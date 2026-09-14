"""万舟こそがエッジの本体か＋wild_scoreがその万舟を事前に当てられるか（ユーザー仮説の検証）。

ユーザー洞察: 「1号が飛んでも万舟が出て、それがROIを引っ張ってる。なら万舟こそが答え」。
検証3本:
  ①万舟がROIを引っ張ってるか: contrarian 3連単/3連複ROIを配当帯で分解(万舟>=1万が本体か)。
  ②wild_scoreが事前に当てるか: wild_score帯別に 1号勝率↓・万舟率↑・平均配当↑ が並ぶか(荒れ捕捉)。
  ③狙う価値があるか(割安か): 高wild帯の確定払戻ROIが>100%か(=万舟が割安=狙い成立)/
     それとも公開の荒れ条件は織り込まれ低wildと変わらんか(=当てても勝てん)。

規律: 確定払戻・クリーン年walk-forward・contrarian(本命≠1号)。万舟=trifecta_yen>=10000。
例: python -u -m src.test_wild
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import storage
from .predict import harville_trifecta
from .wild import score_race
from .strategy_search import trifecta_probs
from .validate import fit_predict_leakfree, SELECTION_YEARS

MANSHU = 10000   # 万舟しきい


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.trio_combo, p.trio_yen, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def trio_rank(p):
    agg = defaultdict(float)
    for c, pr in trifecta_probs(p):
        agg["-".join(sorted(c.split("-"), key=int))] += pr
    return [c for c, _ in sorted(agg.items(), key=lambda x: -x[1])]


def collect(df):
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
            if int(top["lane"]) == 1 or pd.isna(top["trifecta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            srows = [{"lane": int(r.lane), "win_pct": p[int(r.lane)] * 100,
                      "racer_class": r.racer_class} for r in g.itertuples()]
            w = score_race(srows)
            tf_rank = [c for c, _ in sorted(harville_trifecta(p).items(), key=lambda x: -x[1])]
            tr_rank = trio_rank(p)
            rows.append({
                "wild": w["score"],
                "winner": int(top["tansho_lane"]) if not pd.isna(top["tansho_lane"]) else 0,
                "tf_combo": str(top["trifecta_combo"]), "tf_yen": float(top["trifecta_yen"]),
                "tf_hit3": str(top["trifecta_combo"]) in tf_rank[:3],
                "tr_combo": str(top["trio_combo"]), "tr_yen": float(top["trio_yen"]),
                "tr_hit4": str(top["trio_combo"]) in tr_rank[:4],
            })
    return rows


def main():
    df = load()
    rows = collect(df)
    n = len(rows)
    print(f"contrarian(本命≠1号)・クリーン年 {n:,}レース  万舟=3連単>={MANSHU:,}円\n")

    # ①万舟がROIを引っ張るか: 3連単top3のROIを配当帯で分解
    print("【①万舟が利益を引っ張ってるか（3連単top3・確定払戻）】")
    bands = [(0, 3000), (3000, 10000), (10000, 50000), (50000, 10**9)]
    lbl = ["〜3千", "3千〜1万", "万舟1〜5万", "5万〜"]
    stake3 = n * 3 * 100
    for (lo, hi), nm in zip(bands, lbl):
        ret = sum(r["tf_yen"] for r in rows if r["tf_hit3"] and lo <= r["tf_yen"] < hi)
        cnt = sum(1 for r in rows if r["tf_hit3"] and lo <= r["tf_yen"] < hi)
        print(f"   {nm:<10} 的中{cnt:>4}本  この帯だけのROI寄与 {ret/stake3*100:>6.1f}%")
    total3 = sum(r["tf_yen"] for r in rows if r["tf_hit3"]) / stake3
    man3 = sum(r["tf_yen"] for r in rows if r["tf_hit3"] and r["tf_yen"] >= MANSHU) / stake3
    print(f"   → 3連単top3 全体{total3*100:.0f}% のうち 万舟(>={MANSHU})由来 {man3*100:.0f}%"
          f"（{man3/total3*100:.0f}%が万舟から）")

    # ②wild_scoreが荒れ(万舟)を事前に当てるか: score帯別
    print("\n【②wild_scoreが荒れを事前に当ててるか（score四分位）】")
    qs = np.quantile([r["wild"] for r in rows], [0.25, 0.5, 0.75])
    def qband(w):
        return 0 if w < qs[0] else 1 if w < qs[1] else 2 if w < qs[2] else 3
    seg = defaultdict(list)
    for r in rows:
        seg[qband(r["wild"])].append(r)
    print(f"   {'wild帯':<10}{'N':>6}{'1号勝率':>8}{'万舟率':>8}{'平均3連単配当':>12}")
    for q in range(4):
        rs = seg[q]
        if not rs:
            continue
        one = np.mean([r["winner"] == 1 for r in rs])
        man = np.mean([r["tf_yen"] >= MANSHU for r in rs])
        avg = np.mean([r["tf_yen"] for r in rs])
        rng = f"Q{q+1}"
        print(f"   {rng:<10}{len(rs):>6}{one*100:>7.1f}%{man*100:>7.1f}%{avg:>11,.0f}円")

    # ③高wild帯は割安か: 各帯の3連単top3 / 3連複top4 ROI
    print("\n【③高wildレースは割安か（帯別ROI・確定払戻）】")
    print(f"   {'wild帯':<10}{'N':>6}{'3連単top3':>10}{'3連複top4':>10}")
    for q in range(4):
        rs = seg[q]
        if not rs:
            continue
        m = len(rs)
        tf = sum(r["tf_yen"] for r in rs if r["tf_hit3"]) / (m * 3 * 100)
        tr = sum(r["tr_yen"] for r in rs if r["tr_hit4"]) / (m * 4 * 100)
        print(f"   {'Q'+str(q+1):<10}{m:>6}{tf*100:>9.0f}%{tr*100:>9.0f}%")

    print("\n--- 読み ---")
    print("  ①万舟由来の割合が大きい → お前の『万舟こそ本体』は正しい（fat-tailはバグでなく機構）。")
    print("  ②高wild帯で1号勝率↓・万舟率↑が並ぶ → wild_scoreは荒れを事前に捕捉できてる。")
    print("  ③高wild帯のROIが低wildより高い(かつ>100%) → 荒れが割安=狙う価値あり(参考の質が高い)。")
    print("     高wild帯が低wildと大差ない/低い → 公開条件は織り込まれ済=当てても勝てん(=エンタメ止まり)。")
    print("  ※確定払戻ベース=ライブ未証明。儲け主張でなく『スコアが荒れ/万舟を捉えてるか』の裏取り。")


if __name__ == "__main__":
    main()
