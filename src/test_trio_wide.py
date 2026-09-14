"""「荒れそうなら点数を増やす(最大20点=全通り)」が妙味レースで+EVか。

ユーザー案: 荒れて高配当(20倍+)が来そうなら3連複を広く(20点)張ってええやろ？
数学の罠: 普通のレースで全通り買い=パリミュチュエルの控除率(~25%)を確実に負ける(ROI~75%)。
但し妙味レース(本命≠1号)は群衆のイン過剰投票で非イン目が割安→全通りでも>100%で回る"かも"。
  →確定払戻ベースで、点数を4/6/8/12/20と伸ばしROIがどこで100%を割るかを実測する。

規律: クリーン年walk-forward・確定払戻。妙味(本命≠1号)を p0(モデル本命の正規化勝率)帯で分解。
  20点=全通りは必ず的中→ROI=平均trio払戻/2000。>100%なら「割安が控除を上回る」＝広張り成立。

例: python -u -m src.test_trio_wide
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import storage
from .strategy_search import trifecta_probs
from .validate import fit_predict_leakfree, SELECTION_YEARS

POINTS = [1, 4, 6, 8, 12, 20]


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trio_combo, p.trio_yen, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trio_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def trio_rank(p):
    agg = defaultdict(float)
    for combo, prob in trifecta_probs(p):
        agg["-".join(sorted(combo.split("-"), key=int))] += prob
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
            if int(top["lane"]) == 1 or pd.isna(top["trio_yen"]):
                continue                              # 妙味(本命≠1号)のみ
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            rows.append({
                "p0": p[int(top["lane"])],
                "rank": trio_rank(p),
                "combo": str(top["trio_combo"]), "yen": float(top["trio_yen"]),
            })
    return rows


def curve(rows):
    n = len(rows)
    out = {}
    for k in POINTS:
        ret = sum(r["yen"] for r in rows if r["combo"] in r["rank"][:k])
        out[k] = ret / (k * 100 * n)
    return n, out


def main():
    df = load()
    rows = collect(df)
    print(f"妙味(本命≠1号)・クリーン年 {len(rows):,}レース  3連複 点数×ROI\n")

    def band(r):
        if r["p0"] >= 0.50:
            return "堅い妙味(p0≥.50)"
        if r["p0"] >= 0.35:
            return "中(.35-.50)"
        return "薄い/荒れ寄り(p0<.35)"

    segs = defaultdict(list)
    for r in rows:
        segs[band(r)].append(r)

    hdr = "  " + f"{'セグメント':<22}{'N':>6}" + "".join(f"{str(k)+'点':>8}" for k in POINTS)
    print(hdr)
    for lab in ["全妙味"] + sorted(segs):
        rs = rows if lab == "全妙味" else segs[lab]
        if len(rs) < 50:
            continue
        n, c = curve(rs)
        line = f"  {lab:<22}{n:>6}" + "".join(f"{c[k]*100:>7.0f}%" for k in POINTS)
        print(line)

    print("\n--- 読み ---")
    n, c = curve(rows)
    cross = next((k for k in POINTS if c[k] < 1.0), None)
    print(f"  全妙味: 20点(全通り)ROI {c[20]*100:.0f}% ← ここが『広張りの底』。")
    if c[20] >= 1.0:
        print("  ★20点(全通り)でも>100% = 妙味レースは割安が控除を上回る = お前の『荒れたら広く』は成立。")
        print("    ただしROIは点を増やすほど確実に下がる(集中が最効率)。広張りは『当てる快感↑/効率↓』のトレード。")
    else:
        print(f"  ✗20点は{c[20]*100:.0f}%<100% = 全通りは控除負け。100%を割るのは概ね{cross}点あたり。")
        print("    =広げるほど損。『荒れたら広く』は感覚の罠で、効率は『点を絞る』方(1点が最高ROI)。")
    print("\n※これは平均。個別レースで実配当>賭金なら勝つが、事前に『どの1点が来るか』は当てられん(分かれば1点張り)。")
    print("※確定払戻ベース=ライブ未証明。トリオodズの前向き収集をすればEV版(高EVだけ広く)が組める=次の一手。")


if __name__ == "__main__":
    main()
