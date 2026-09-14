"""荒れ読みレースで3連単 vs 3連複どちらが良い器か。

動機: 2連単/3連単の分解(#1・experiential⑤)で「2-3着の"順序"スキルはゼロ=belief≒random」。
  なら順序を要求する3連単は当たらん部分に賭けてる。3連複(着順不問の上位3艇セット)なら
  順序を捨て「どの3艇が絡むか」だけに賭けられる=エッジが在るセット選択だけを
  120通り→20通りの低希釈・低分散で獲れるかもしれん。荒れ検知が効く群でこそ生きる。

設計: HGBモデル(確立パイプライン)・クリーン年walk-forward・本命≠1号艇(荒れ読み)レース。
  同じレース群で 3連単top-N と 3連複top-N の回収率を並べ、fat-tail頑健性も比較。
  順序スキルが無いなら 3連複ROI ≥ 3連単ROI かつ tail依存が軽い、が期待。

規律: ROIは確定払戻(trio_yen/trifecta_yen)。全張りbaselineを刻む。壁(控除25%)を忘れん。

例: python -u -m src.test_trio
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import storage
from .features import FEATURES, build_frame          # noqa: F401 (fit_predict_leakfreeが使用)
from .strategy_search import trifecta_probs
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.trio_combo, p.trio_yen,
               p.exacta_combo, p.exacta_yen, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trio_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def trio_probs(p):
    """3連複セット確率 = 6通りの着順(Harville)を合算。降順list[(sorted_combo, prob)]。"""
    agg = defaultdict(float)
    for combo, prob in trifecta_probs(p):
        key = "-".join(sorted(combo.split("-"), key=int))
        agg[key] += prob
    return sorted(agg.items(), key=lambda x: -x[1])


def collect(df):
    """クリーン年walk-forward・本命≠1号艇レースを1件化(3単/3複の確率rank・実結果・配当)。"""
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for _, g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1:                     # 荒れ読み=本命≠1号艇のみ
                continue
            if pd.isna(top["trifecta_yen"]) or pd.isna(top["trio_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            # 2連単 i-j ランク（アンカー用）
            ex = []
            for i in p:
                di = 1 - p[i]
                if di > 0:
                    for j in p:
                        if j != i:
                            ex.append((f"{i}-{j}", p[i] * p[j] / di))
            ex.sort(key=lambda x: -x[1])
            won_t = (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == int(top["lane"])
            rows.append({
                "p": p,                                    # 勝率dict{lane:p}（セット分解用）
                "tf_rank": [c for c, _ in trifecta_probs(p)],
                "tf_combo": str(top["trifecta_combo"]), "tf_yen": float(top["trifecta_yen"]),
                "tr_rank": [c for c, _ in trio_probs(p)],
                "tr_combo": str(top["trio_combo"]), "tr_yen": float(top["trio_yen"]),
                "ex_rank": [c for c, _ in ex],
                "ex_combo": str(top["exacta_combo"]), "ex_yen": float(top["exacta_yen"]) if not pd.isna(top["exacta_yen"]) else 0.0,
                "won_t": won_t, "tan_yen": float(top["tansho_yen"]) if not pd.isna(top["tansho_yen"]) else 0.0,
            })
    return rows


def roi(rows, rank_key, combo_key, yen_key, k):
    ret = hit = 0
    for r in rows:
        if r[combo_key] in r[rank_key][:k]:
            hit += 1
            ret += r[yen_key]
    n = len(rows)
    return hit / n, ret / (k * 100 * n)


def fat_tail(rows, rank_key, combo_key, yen_key, k):
    rets = np.array([r[yen_key] if r[combo_key] in r[rank_key][:k] else 0.0 for r in rows])
    rets = np.sort(rets)[::-1]
    stake = len(rows) * k * 100
    return rets, stake


def main():
    df = load()
    rows = collect(df)
    n = len(rows)
    print(f"荒れ読み(本命≠1号艇)・クリーン年 {n:,}レース  [3連単 vs 3連複]\n")

    # ===== アンカー: 既知の確立値(単勝~116%/2連単3点~188%)を同ハーネスで再現するか =====
    tan_hit = sum(r["won_t"] for r in rows)
    tan_roi = sum(r["tan_yen"] for r in rows if r["won_t"]) / (n * 100)
    ex_hit = ex_ret = 0
    for r in rows:
        if r["ex_combo"] in r["ex_rank"][:3]:
            ex_hit += 1
            ex_ret += r["ex_yen"]
    ex_roi = ex_ret / (3 * 100 * n)
    print("  【アンカー照合】確立値と一致すればハーネス健全→3系も同じ土俵")
    print(f"   単勝(本命1点):   的中{tan_hit/n*100:.1f}%  ROI {tan_roi*100:.1f}%   (既知~115-118%)")
    print(f"   2連単(上位3点):  的中{ex_hit/n*100:.1f}%  ROI {ex_roi*100:.1f}%   (既知~183-195%)")
    anchor_ok = 108 <= tan_roi * 100 <= 125 and 170 <= ex_roi * 100 <= 205
    print(f"   -> {'[OK] 整合 3系の数字も信用してよい' if anchor_ok else '[!] ズレ ハーネス要調査(3系の数字は保留)'}\n")

    print(f"  {'点数':>4}  {'3連単的中':>9}{'3連単ROI':>9}   {'3連複的中':>9}{'3連複ROI':>9}")
    for k in [1, 2, 3, 4, 6]:
        th, tr_roi = roi(rows, "tf_rank", "tf_combo", "tf_yen", k)
        sh, sr_roi = roi(rows, "tr_rank", "tr_combo", "tr_yen", k)
        print(f"  {k:>4}  {th*100:>8.1f}%{tr_roi*100:>8.1f}%   {sh*100:>8.1f}%{sr_roi*100:>8.1f}%")

    # fat-tail比較: 各賭式の"最良ROIっぽいN"で上位除外
    print("\n  【fat-tail頑健性比較】上位本除外でROIがどこまで落ちるか")
    for label, rk, ck, yk, k in [("3連単6点", "tf_rank", "tf_combo", "tf_yen", 6),
                                  ("3連複3点", "tr_rank", "tr_combo", "tr_yen", 3),
                                  ("3連複4点", "tr_rank", "tr_combo", "tr_yen", 4)]:
        rets, stake = fat_tail(rows, rk, ck, yk, k)
        hits = int((rets > 0).sum())
        line = " ".join(f"{x}本→{rets[x:].sum()/stake*100:.0f}%" for x in [0, 1, 3, 10, 30])
        print(f"   {label}(的中{hits}/最大{int(rets[0]):,}円): {line}")

    # ===== 裏取り: セット選択スキルの#1式3-way分解（3連複を1点だけ張る）=====
    def set_key(lanes):
        return "-".join(sorted((str(int(x)) for x in lanes), key=int))

    rng = np.random.default_rng(0)

    def set_roi(mode, reps=1):
        rois, hits = [], []
        for _ in range(reps):
            ret = hit = 0
            for r in rows:
                p = r["p"]
                order = sorted(p, key=lambda l: -p[l])
                if mode == "model":
                    s = order[:3]
                elif mode == "fix1_rand2":                # 1着call固定＋相方2艇ランダム
                    s = [order[0]] + list(rng.choice(order[1:], 2, replace=False))
                else:                                     # 完全ランダム3艇
                    s = list(rng.choice(order, 3, replace=False))
                if set_key(s) == r["tr_combo"]:
                    hit += 1
                    ret += r["tr_yen"]
            rois.append(ret / (100 * len(rows)))
            hits.append(hit / len(rows))
        return np.mean(rois), np.mean(hits)

    m_roi, m_hit = set_roi("model")
    f_roi, f_hit = set_roi("fix1_rand2", reps=10)
    r_roi, r_hit = set_roi("rand3", reps=10)
    print("\n  【裏取り: セット選択スキル 3-way分解（3連複1点張り・同一レース）】")
    print(f"   model(上位3艇)      : 的中{m_hit*100:>5.1f}%  ROI {m_roi*100:>6.1f}%")
    print(f"   fix1+rand2(1着call固定): 的中{f_hit*100:>5.1f}%  ROI {f_roi*100:>6.1f}%")
    print(f"   rand3(完全ランダム)  : 的中{r_hit*100:>5.1f}%  ROI {r_roi*100:>6.1f}%")
    companion = m_roi > f_roi * 1.15
    call = f_roi > r_roi * 1.15
    if companion and call:
        print("   → model≫fix1≫rand = 1着call＋相方2艇の両方に本物スキル(セット選択は実力)")
    elif call:
        print("   → model≒fix1≫rand = 1着contrarian callが本体・相方2艇は無スキル(#1と同型)")
    else:
        print("   → 差小 = セット選択スキルは薄い・ROIは分散/配当構造寄り")

    print("\n※読み: 順序スキル無しが本当なら 3連複ROI≧3連単ROI かつ 3連複の方がtail依存が軽い。")
    print("  それでも全て<100%なら=荒れ読みでも3系は控除の壁。獲るなら既検証の単勝/2連単が素直。")
    print("  3連複が単独で>100%かつtail頑健なら=順序を捨てた分だけ器として優秀=新しい実用候補。")


if __name__ == "__main__":
    main()
