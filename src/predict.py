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

# ── 着順展開の段階別指数（Lo–Bacon-Shone型）────────────────────────────────
# 素のHarvilleは「1着の確率ベクトルが2着・3着もそのまま支配する」と仮定するが、
# 実測ではその仮定が成立せん。OOS 3,350レース(20260716-20260806/model.joblib=7/15学習)で
# 最尤当てはめ＋ブートストラップ200回:
#     γ (1着の鋭さ)   = 1.25  95%CI [1.20, 1.30]
#     λ2(2着の減衰)   = 0.60  95%CI [0.60, 0.65]
#     λ3(3着の減衰)   = 0.40  95%CI [0.40, 0.45]
# 3つとも95%CIが1.00を外す＝素のHarville(全部1.00)は棄却される。
# 症状: 台帳(本命≠1号艇の667件)で 3連複4点=予測72.2%/実際52.9%、3連単3点=予測21.8%/実際10.8%。
#       2着以降ほど実際はバラけるのに、素のHarvilleは上位艇へ確率を寄せすぎとった。
# 再当てはめ: python -m src.fit_stage_exponents
STAGE_GAMMA = 1.25    # 1着: p ∝ p^γ (γ>1 = モデルは1着側で控えめすぎた分を鋭くする)
STAGE_LAMBDA2 = 0.60  # 2着: ∝ p^λ2 (λ2<1 = 平たくする)
STAGE_LAMBDA3 = 0.40  # 3着: ∝ p^λ3 (さらに平たい)


def calibrated_win(win_prob: dict[int, float], gamma: float | None = None) -> dict[int, float]:
    """レース内で正規化し、較正指数γで鋭くしたP(1着)を返す（合計1）。

    gamma=None なら STAGE_GAMMA を都度読む（定数を差し替えた検証がそのまま効くように、
    既定引数で定義時に固定せん）。
    """
    gamma = STAGE_GAMMA if gamma is None else gamma
    s = sum(win_prob.values())
    if s <= 0:
        n = len(win_prob) or 1
        return {k: 1 / n for k in win_prob}
    w = {k: (v / s) ** gamma for k, v in win_prob.items()}
    t = sum(w.values()) or 1
    return {k: v / t for k, v in w.items()}


def _stage(p: dict[int, float], taken: tuple[int, ...], lam: float) -> dict[int, float]:
    """既に着順が決まった艇(taken)を除いた残りの、次の着の確率分布（合計1）。

    素のHarvilleは lam=1.0（残りをそのままの比で配る）。lam<1 で平たくする。
    """
    w = {k: p[k] ** lam for k in p if k not in taken}
    s = sum(w.values())
    if s <= 0:
        n = len(w) or 1
        return {k: 1 / n for k in w}
    return {k: v / s for k, v in w.items()}


def exacta_probs(win_prob: dict[int, float]) -> dict[str, float]:
    """各艇のP(1着)から2連単 i-j の確率を推定する（段階別指数つき）。"""
    p = calibrated_win(win_prob)
    out: dict[str, float] = {}
    for i in p:
        second = _stage(p, (i,), STAGE_LAMBDA2)
        for j, pj in second.items():
            out[f"{i}-{j}"] = p[i] * pj
    return out


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

    # 較正済みP(1着): レース内で合計1にした上で指数γで鋭くした値。EV・買い目の土台はこっち。
    # win_pct(生の正規化)は katai/papertrade/sensor/streamlit の閾値が乗っとるので
    # スケールを動かさず据え置き。ただし実測では控えめ側にズレる(台帳: 予測36.1%/実際42.1%)。
    cal = calibrated_win({e["lane"]: float(raw[i]) for i, e in enumerate(entries)})

    out = []
    for i, e in enumerate(entries):
        r = rank_of[i]
        out.append({
            **e,
            "win_prob": round(float(raw[i]), 4),   # 分類器の生出力(レース内合計は1にならん)
            "win_pct": round(float(win_pct[i]), 1),  # レース内正規化(表示用・既存閾値の基準)
            "win_p": round(cal[e["lane"]], 4),       # 較正済みP(1着)。合計1。EV用
            "win_pct_cal": round(100 * cal[e["lane"]], 1),  # 同上を%表示にしたもの
            "rank": r + 1,
            "mark": MARKS[r] if r < len(MARKS) else " ",
        })
    out.sort(key=lambda x: x["rank"])
    return out


def attach_odds(rows: list[dict], odds: dict[int, float] | None) -> list[dict]:
    """各艇にオッズと期待値(EV = P(1着) × 単勝オッズ)を付与。

    EV > 1.0 が理論上プラス（割安）。odds が無ければ None のまま。

    【EVに使う確率】win_prob は二値分類器の生出力で、レース内6艇の合計が1にならん
    （実測 中央1.40・5-95%点1.13-1.68 / n=13,417スナップショット）。これを直接
    オッズに掛けると EV が中央で4割ほど過大に出て、EV>1.0 の「割安」が実際には
    真のEV 0.7台で発火してまう。よってレース内で正規化＋較正した P(1着) を使う。
    """
    p = calibrated_win({e["lane"]: e["win_prob"] for e in rows})
    for e in rows:
        o = odds.get(e["lane"]) if odds else None
        pe = e.get("win_p") or p.get(e["lane"], 0.0)   # predict_entries が付けとればそれを使う
        e["win_p"] = round(pe, 4)              # レース内で合計1のP(1着)。EVの土台
        e["odds"] = o
        e["ev"] = round(pe * o, 2) if o else None
        e["value"] = bool(o and pe * o > 1.0)
    return rows


def recommend(rows: list[dict]) -> dict:
    """検証済みの「勝てる形」に基づく推奨買い目を返す。

    戦略(docs/FINDINGS.md): モデル本命がイン(1号艇)以外のレースだけ、
    その本命を単勝で買う（回収率の主張はしない=前向き検証を通していないため）。
    2連単・確率上位3点も有望(165-171%)。本命=1号艇は妙味薄＝見送り。
    rows は predict_entries の出力（rank昇順, win_prob 付き）。
    """
    if not rows:
        return {"bet": False, "reason": "出走表なし"}
    honmei = rows[0]  # rank=1
    if honmei["lane"] == 1:
        return {"bet": False,
                "reason": "モデル本命がイン(1号艇)＝公衆と同じ見立て。妙味薄のため見送り推奨。"}

    # 2連単 上位3点（段階別指数つき: exacta_probs）
    ex = sorted(exacta_probs({e["lane"]: e["win_prob"] for e in rows}).items(),
                key=lambda x: -x[1])

    # 3連複 上位4点（着順不問セット。順序スキル0を回避しセット選択だけ使う頑健な器。
    #  test_trio: 荒れ読みで4点258%/上位30本抜き222%とfat-tail頑健。ROI源はレース選択）
    trio = sorted(trio_probs({e["lane"]: e["win_prob"] for e in rows}).items(),
                  key=lambda x: -x[1])
    # 3連単 上位3点（着順あり・大きい方＝万舟狙い。fat-tailで脆い・
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
        "reason": "モデルがイン(1号艇)を否定＝インバイアスの妙味。位置(#1)が崩れれば地力(実力=#2)が勝者を決める"
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
    """各艇の1着確率(win_prob)から3連単 i-j-k の確率を推定する。

    素のHarvilleは P(i-j-k) = p_i · p_j/(1-p_i) · p_k/(1-p_i-p_j) やが、
    実測で2着・3着ほど確率がバラける（STAGE_LAMBDA2/3 の由来コメント参照）ので
    段階ごとに指数で平たくする:
        P(i-j-k) = q_i · (q_j^λ2 / Σ) · (q_k^λ3 / Σ)     q = calibrated_win(p)
    λ2=λ3=γ=1.0 に戻せば素のHarvilleと一致する。
    """
    if sum(win_prob.values()) <= 0:
        return {}
    p = calibrated_win(win_prob)
    out: dict[str, float] = {}
    for i in p:
        second = _stage(p, (i,), STAGE_LAMBDA2)
        for j, pj in second.items():
            third = _stage(p, (i, j), STAGE_LAMBDA3)
            for k, pk in third.items():
                out[f"{i}-{j}-{k}"] = p[i] * pj * pk
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

# 💎高確信フラグ: 本命≠1号 × 本命勝率≥0.45 × 2連単top3(確率順)EV>3.5（3条件が揃った最濃断面）
STRONGEST_P0 = 0.45
STRONGEST_EV = 3.5


def dynamic_buy(rows: list[dict], odds: dict, th: float = DYNAMIC_EV_TH, gate: dict | None = None) -> dict:
    """モデル確率×締切オッズで各賭式の全目のEVを出し、EV>th の割安だけ選抜（動的点数）。

    odds = {"tansho":{lane:odズ}, "exacta":{'i-j':odズ}, "trio":{'a-b-c':odズ}, "trifecta":{'i-j-k':odズ}}。
    割安が多いレース→点数増（＝万舟つく→広く）、少ない→絞る＝『EVで点数を自己調整』の心臓。
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
