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

# 高確信妙味の閾値: 本命の正規化勝率(win_pct)がこれ以上なら🔥フラグ（強調表示のみ）
HIGH_CONF_PCT = 50.0


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

    # 3連複 上位4点（着順不問セット。順序スキル0を回避しセット選択だけ使う頑健な器。
    #  test_trio: 荒れ読みで4点258%/上位30本抜き222%とfat-tail頑健。ROI源はレース選択）
    trio = sorted(trio_probs({e["lane"]: e["win_prob"] for e in rows}).items(),
                  key=lambda x: -x[1])
    # 3連単 上位3点（着順あり・大きい方＝万舟狙い。test_trio: 荒れ読み3点533%だがfat-tail脆い・
    #  着順スキル0で高ROIは配当構造由来。別枠track・アクセル踏み込み用）
    tf = sorted(harville_trifecta({e["lane"]: e["win_prob"] for e in rows}).items(),
                key=lambda x: -x[1])

    # 高確信妙味フラグ: 本命の正規化勝率(p0)≥50%。test_trio_only で妙味×p0≥.50 が
    #  最ジューシー帯（N325・fat-tailで水準は不明）→ 強調表示だけ。ROI水準は主張しない。
    conf = honmei.get("win_pct")
    high_conf = bool(conf is not None and conf >= HIGH_CONF_PCT)

    # 【#2の答え=地力(実力)】インが崩れた世界(本命≠1号)で勝者を決めるのは選手の地力。
    #  test_when_inside_fails: 非イン勝者を当てる力=地力(全国勝率)38.6% > 当地34.9 > 展示/出足31
    #  > 2号ベタ(差し定番)30.7 > 機力23.9。位置(#1)の次に効く柱は"実力"。定番の差し2号でも速さでもない。
    #  ∴妙味の本命に地力ランクを添え、実力の裏付けthaある本命か(実力型)を示す＋インが飛んだ時の実力対抗も出す。
    nat = {e["lane"]: (e.get("nat_win") or 0.0) for e in rows}
    power_order = sorted(nat, key=lambda l: -nat[l])            # 地力(全国勝率)降順の枠
    power_rank = {l: i + 1 for i, l in enumerate(power_order)}  # 1=6人中の地力最強
    honmei_power_rank = power_rank.get(honmei["lane"])
    power_honmei = bool(honmei_power_rank is not None and honmei_power_rank <= 2)  # 💪実力型本命
    rival_lanes = [l for l in power_order if l != 1][:1]        # 非イン地力トップ=実力対抗
    power_rival = rival_lanes[0] if rival_lanes else None
    rival_row = next((e for e in rows if e["lane"] == power_rival), None)

    # ○対抗=地力トップの"非イン・非本命"艇。買い目に組込む:イン(#1)が崩れた時、勝者を決めるのは
    #  実力(#2=地力38.6%>差し2号30.7)。本命が実力筋でも、対抗として実力2番手を必ず拾う。
    taiko = next((l for l in power_order if l != 1 and l != honmei["lane"]), None)
    taiko_row = next((e for e in rows if e["lane"] == taiko), None)
    # 実力筋2連単: 本命⇄対抗(位置が崩れた世界の実力ワンツー)。既存exacta3(Harville)に無ければ足す
    jitsuryoku_ex = []
    if taiko is not None:
        jitsuryoku_ex = [f"{honmei['lane']}-{taiko}", f"{taiko}-{honmei['lane']}"]
    exacta_buy = [c for c, _ in ex[:3]]
    for c in jitsuryoku_ex:                                     # 実力ワンツーを買い目に確実に含める
        if c not in exacta_buy:
            exacta_buy.append(c)

    return {
        "bet": True,
        "tansho": honmei["lane"],
        "tansho_name": honmei.get("name"),
        "conf": conf,                                       # 本命の正規化勝率(表示用)
        "high_conf": high_conf,                             # 🔥高確信妙味フラグ
        "exacta3": [c for c, _ in ex[:3]],
        "exacta3_p": [round(pr, 5) for _, pr in ex[:3]],  # EV算出用: 各組のHarville確率
        "trio4": [c for c, _ in trio[:4]],
        "trio4_p": [round(pr, 5) for _, pr in trio[:4]],   # 3連複セット確率(表示/EV/追跡用)
        "trifecta3": [c for c, _ in tf[:3]],               # 3連単上位3点(着順あり・大きい方)
        "trifecta3_p": [round(pr, 5) for _, pr in tf[:3]],
        "trio_rank": [c for c, _ in trio],                 # 3連複全ランク(買い方くらべ用)
        "trifecta_rank": [c for c, _ in tf],               # 3連単全ランク(買い方くらべ用)
        "power_rank": honmei_power_rank,                    # 本命の地力ランク(1=6人中の実力最強)
        "power_honmei": power_honmei,                       # 💪実力型本命(地力top2=位置が崩れても実力で来る型)
        "power_rival": power_rival,                         # インが崩れた時の実力対抗(非イン地力トップ枠)
        "power_rival_name": rival_row.get("name") if rival_row else None,
        "taiko": taiko,                                     # ○対抗=地力トップの非イン非本命艇(#2=実力)
        "taiko_name": taiko_row.get("name") if taiko_row else None,
        "power_ex": jitsuryoku_ex,                          # 実力筋2連単(本命⇄対抗)=位置崩れの実力ワンツー
        "reason": "モデルがイン(1号艇)を否定＝インバイアスの妙味。位置(#1)が崩れれば地力(実力=#2)that勝者を決める"
                  f"(検証: 非イン勝者の最有力軸=全国勝率38.6%>差し2号30.7)。本命の地力{honmei_power_rank}位/6。"
                  "※確定払戻ベース・全券種ライブ再現(6艇)では控除の壁=エッジ主張はしない。",
    }


# 2連単EVフィルタ閾値（test_ev_probe: EV>2.0帯がfat-tail生存で堅く300-500%・CI下限>100%）
EXACTA_EV_THRESHOLD = 2.0


def exacta_ev(exacta3, exacta3_p, live_odds):
    """2連単EV = Σ(p_c × 締切odds_c)/3。live_odds={'i-j':odズ}。全組そろわねば None。

    test_ev_probe と同一定義。EV>EXACTA_EV_THRESHOLD が「買い」。
    """
    if not exacta3 or not exacta3_p or not live_odds:
        return None
    ods = [live_odds.get(c) for c in exacta3]
    if any(o is None or o <= 0 for o in ods):
        return None
    return sum(p * o for p, o in zip(exacta3_p, ods)) / len(exacta3)


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


def trio_probs(win_prob: dict[int, float]) -> dict[str, float]:
    """各艇の1着確率→3連複(着順不問)セット確率。

    harville_trifecta の6順列(i-j-k)を無順序セットへ合算。キーは 'a-b-c' (a<b<c)。
    test_trio.trio_probs と同一定義（荒れ読みで頑健な器と実証した集約）。
    """
    from collections import defaultdict
    agg: dict[str, float] = defaultdict(float)
    for combo, p in harville_trifecta(win_prob).items():
        key = "-".join(sorted(combo.split("-"), key=int))
        agg[key] += p
    return dict(agg)


def trio_ev(rows: list[dict], odds: dict[str, float]) -> list[dict]:
    """3連複の各セットについて EV = 確率 × オッズ を計算し、EV降順で返す。

    rows は predict_entries の出力（win_prob を含む）。odds は fetch_trio_odds。
    ※3連複エッジの本体は"荒れレースの選択"（本命≠1号艇）で、EVは表示/追跡用。
      2連単のような検証済みEV閾値はまだ無い（確定払戻ベースの器・ライブ未証明）。
    """
    probs = trio_probs({e["lane"]: e["win_prob"] for e in rows})
    res = []
    for combo, p in probs.items():
        o = odds.get(combo)
        if o is None:
            continue
        res.append({"combo": combo, "prob": round(p, 4),
                    "odds": o, "ev": round(p * o, 2)})
    res.sort(key=lambda x: -x["ev"])
    return res


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


# 動的買い目の割安ライン。2連単EV>2.0は検証済み。3連系は実験（3連単ライブ収集で調整予定）。
DYNAMIC_EV_TH = 2.0

# 💎最強妙味: 本命≠1号 × 本命勝率≥0.45 × 2連単top3(確率順)EV>3.5（backtest 592%/全年/頑健の最濃断面）
STRONGEST_P0 = 0.45
STRONGEST_EV = 3.5


def dynamic_buy(rows: list[dict], odds: dict, th: float = DYNAMIC_EV_TH, gate: dict | None = None) -> dict:
    """モデル確率×締切オッズで各賭式の全目のEVを出し、EV>th の割安だけ選抜（動的点数）。

    odds = {"tansho":{lane:odズ}, "exacta":{'i-j':odズ}, "trio":{'a-b-c':odズ}, "trifecta":{'i-j-k':odズ}}。
    割安that多いレース→点数増（＝万舟つく→広く）、少ない→絞る＝『EVで点数を自己調整』の心臓。
    gate=sensor.bet_gate()：ok=Falseの賭式(＝持続センサーがDECAYED)は自動でOFF＝ライブ劣化を反映。
    """
    wp = {e["lane"]: e["win_prob"] for e in rows}
    s = sum(wp.values()) or 1
    p = {k: v / s for k, v in wp.items()}

    def ev_list(prob_map, od):
        out = [{"combo": c, "p": round(pr, 4), "odds": od[c], "ev": round(pr * od[c], 2)}
               for c, pr in prob_map.items() if od and od.get(c)]
        return sorted(out, key=lambda x: -x["ev"])

    tan_p = {str(l): p[l] for l in p}
    exa_p = {}
    for i in p:
        di = 1 - p[i]
        if di > 0:
            for j in p:
                if j != i:
                    exa_p[f"{i}-{j}"] = p[i] * p[j] / di
    allb = {
        "tansho": ev_list(tan_p, odds.get("tansho") and {str(k): v for k, v in odds["tansho"].items()}),
        "exacta": ev_list(exa_p, odds.get("exacta")),
        "trio": ev_list(trio_probs(wp), odds.get("trio")),
        "trifecta": ev_list(harville_trifecta(wp), odds.get("trifecta")),
    }
    buy, off = {}, []
    for k, v in allb.items():
        if gate and not gate.get(k, {}).get("ok", True):
            buy[k] = []                      # 持続センサーがDECAYED→この賭式OFF
            off.append(k)
        else:
            buy[k] = [x for x in v if x["ev"] > th]
    pts = sum(len(v) for v in buy.values())
    # 💎最強妙味: 本命≠1号 × p0≥.45 × 2連単top3(確率順)avg EV>3.5
    honmei = max(p, key=p.get)
    p0 = p[honmei]
    ex3 = sorted(exa_p.items(), key=lambda x: -x[1])[:3]
    exo = odds.get("exacta") or {}
    ex_ev3 = (sum(pr * exo[c] for c, pr in ex3) / 3.0) if ex3 and all(exo.get(c) for c, _ in ex3) else None
    strongest = (honmei != 1 and p0 >= STRONGEST_P0 and ex_ev3 is not None and ex_ev3 > STRONGEST_EV)
    return {"buy": buy, "all": allb, "th": th, "points": pts, "cost": pts * 100, "off": off,
            "p0": p0, "ex_ev3": ex_ev3, "strongest": strongest}


def dynamic_for_race(date: str, jcd: str, rno: int, bundle=None,
                     th: float = DYNAMIC_EV_TH, gate: dict | None = None):
    """指定レースをライブ予測＋全賭式の締切オッズ取得→動的買い目。(rows, dynamic) を返す。"""
    rows = predict_race(date, jcd, rno, bundle)
    if not rows:
        return None
    odds = {
        "tansho": {e["lane"]: e["odds"] for e in rows if e.get("odds")},
        "exacta": scraper.fetch_exacta_odds(date, jcd, rno) or {},
        "trio": scraper.fetch_trio_odds(date, jcd, rno) or {},
        "trifecta": scraper.fetch_trifecta_odds(date, jcd, rno) or {},
    }
    return rows, dynamic_buy(rows, odds, th, gate)


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
