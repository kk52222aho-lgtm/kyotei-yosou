"""予測ロジック。

学習済みモデルを使い、指定レースの出走表をライブ取得して各艇の1着確率を算出。
レース内で正規化し、本命(◎)/対抗(○)/単穴(▲) のランクを付与する。
"""
from __future__ import annotations

import os

import joblib
import numpy as np

from . import scraper, storage
from .features import build_frame

MODEL_PATH = os.path.join(storage.DATA_DIR, "model.joblib")

MARKS = ["◎", "○", "▲", "△", "×", " "]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_entries(entries: list[dict], bundle=None) -> list[dict]:
    """出走表 dict のリストに、各艇の勝率予測とランク・印を付与して返す。"""
    bundle = bundle or load_model()
    if bundle is None:
        raise RuntimeError("モデル未学習です。先に python -m src.train を実行してください。")

    model, feats = bundle["model"], bundle["features"]
    frame = build_frame(entries)
    X = frame[feats].to_numpy(dtype=float)
    raw = model.predict_proba(X)[:, 1]

    # レース内で相対正規化（合計100%）
    total = raw.sum()
    win_pct = (raw / total * 100) if total > 0 else np.full(len(raw), 100 / len(raw))

    order = np.argsort(-win_pct)  # 確率降順
    rank_of = {idx: r for r, idx in enumerate(order)}

    out = []
    for i, e in enumerate(entries):
        r = rank_of[i]
        out.append({
            **e,
            "win_prob": round(float(raw[i]), 4),   # 校正済みP(1着)。期待値計算用
            "win_pct": round(float(win_pct[i]), 1),  # レース内正規化(表示用)
            "rank": r + 1,
            "mark": MARKS[r] if r < len(MARKS) else " ",
        })
    out.sort(key=lambda x: x["rank"])
    return out


def attach_odds(rows: list[dict], odds: dict[int, float] | None) -> list[dict]:
    """各艇にオッズと期待値(EV = 校正済み勝率 × 単勝オッズ)を付与。

    EV > 1.0 が理論上プラス（割安）。odds が無ければ None のまま。
    """
    for e in rows:
        o = odds.get(e["lane"]) if odds else None
        e["odds"] = o
        e["ev"] = round(e["win_prob"] * o, 2) if o else None
        e["value"] = bool(o and e["win_prob"] * o > 1.0)
    return rows


def recommend(rows: list[dict]) -> dict:
    """検証済みの「勝てる形」に基づく推奨買い目を返す。

    戦略(docs/FINDINGS.md): モデル本命がイン(1号艇)以外のレースだけ、
    その本命を単勝で買う（大サンプル116.5% / 4期間106-118%, 低分散）。
    2連単・確率上位3点も有望(165-171%)。本命=1号艇は妙味薄＝見送り。
    rows は predict_entries の出力（rank昇順, win_prob 付き）。
    """
    if not rows:
        return {"bet": False, "reason": "出走表なし"}
    honmei = rows[0]  # rank=1
    if honmei["lane"] == 1:
        return {"bet": False,
                "reason": "モデル本命がイン(1号艇)＝公衆と同じ見立て。妙味薄のため見送り推奨。"}

    # 2連単 上位3点（Harville: P(i-j)=p_i·p_j/(1-p_i)）
    p = {e["lane"]: e["win_prob"] for e in rows}
    s = sum(p.values()) or 1
    p = {k: v / s for k, v in p.items()}
    ex = []
    for i in p:
        di = 1 - p[i]
        if di <= 0:
            continue
        for j in p:
            if j != i:
                ex.append((f"{i}-{j}", p[i] * p[j] / di))
    ex.sort(key=lambda x: -x[1])

    return {
        "bet": True,
        "tansho": honmei["lane"],
        "tansho_name": honmei.get("name"),
        "exacta3": [c for c, _ in ex[:3]],
        "reason": "モデルがイン(1号艇)を否定＝インバイアスの妙味。検証で単勝116.5%・2連単上位3点171%。",
    }


def harville_trifecta(win_prob: dict[int, float]) -> dict[str, float]:
    """各艇の1着確率(win_prob)から、Harville(Plackett-Luce)法で
    3連単 i-j-k の確率を推定する。

    P(i-j-k) = p_i · p_j/(1-p_i) · p_k/(1-p_i-p_j)
    win_prob はレース内で合計1に正規化して使う。
    """
    s = sum(win_prob.values())
    if s <= 0:
        return {}
    p = {k: v / s for k, v in win_prob.items()}
    lanes = list(p.keys())
    out: dict[str, float] = {}
    for i in lanes:
        di = 1 - p[i]
        if di <= 0:
            continue
        for j in lanes:
            if j == i:
                continue
            dij = 1 - p[i] - p[j]
            if dij <= 0:
                continue
            for k in lanes:
                if k in (i, j):
                    continue
                out[f"{i}-{j}-{k}"] = p[i] * (p[j] / di) * (p[k] / dij)
    return out


def trifecta_ev(rows: list[dict], odds: dict[str, float]) -> list[dict]:
    """3連単の各買い目について EV = 確率 × オッズ を計算し、EV降順で返す。

    rows は predict_entries の出力（win_prob を含む）。odds は fetch_trifecta_odds。
    """
    win_prob = {e["lane"]: e["win_prob"] for e in rows}
    probs = harville_trifecta(win_prob)
    res = []
    for combo, p in probs.items():
        o = odds.get(combo)
        if o is None:
            continue
        res.append({"combo": combo, "prob": round(p, 4),
                    "odds": o, "ev": round(p * o, 2)})
    res.sort(key=lambda x: -x["ev"])
    return res


def predict_trifecta(date: str, jcd: str, rno: int, bundle=None, top: int = 10):
    """指定レースの3連単 期待値ランキング(上位top)を返す。(rows, ev_list)"""
    rows = predict_race(date, jcd, rno, bundle)
    if not rows:
        return None
    odds = scraper.fetch_trifecta_odds(date, jcd, rno)
    if not odds:
        return rows, []
    return rows, trifecta_ev(rows, odds)[:top]


def predict_race(date: str, jcd: str, rno: int, bundle=None) -> list[dict] | None:
    entries = scraper.fetch_racelist(date, jcd, rno)
    if not entries:
        return None
    for e in entries:
        e["jcd"] = str(jcd).zfill(2)  # 場コードを特徴量(venue_code)へ
    # 直前情報があればマージ（未発表なら欠損のまま＝学習時の中央値で補完）
    before = scraper.fetch_beforeinfo(date, jcd, rno)
    if before:
        per_lane, race_cond = before
        for e in entries:
            bl = per_lane.get(e["lane"], {})
            e.update({k: bl.get(k) for k in
                      ("tenji_time", "tilt", "weight_today", "tenji_st")})
            e.update(race_cond)
    rows = predict_entries(entries, bundle)
    # 単勝オッズを取得して期待値を付与（未確定なら None のまま）
    odds = scraper.fetch_odds(date, jcd, rno)
    return attach_odds(rows, odds)
