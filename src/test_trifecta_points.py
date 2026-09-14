"""過去3年（妙味・確定払戻）で3連単の点数を細かく刻み、効率(回収率)ピークを特定。

test_styles は 1/3/20/60/120 と粗かった。ここは 1..120 を細かく刻んで
「何点が一番効率(ROI)いいか」を出す。総収支・的中率・万舟捕捉も併記
（効率ピークと総額ピークは別、を数字で見る用）。妙味=本命≠1号・クリーン年・1点100円。

例: python -u -m src.test_trifecta_points
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .predict import harville_trifecta
from .validate import fit_predict_leakfree, SELECTION_YEARS

GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 30, 40, 50, 60, 80, 100, 120]
MANSHU = 10000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


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
            rank = [c for c, _ in sorted(harville_trifecta(p).items(), key=lambda x: -x[1])]
            combo = str(top["trifecta_combo"])
            rk = rank.index(combo) + 1 if combo in rank else 999
            rows.append({"rk": rk, "yen": float(top["trifecta_yen"])})
    return rows


def main():
    df = load()
    rows = collect(df)
    n = len(rows)
    print(f"妙味(本命≠1号)・クリーン年 {n:,}レース・1点100円・3連単\n")
    print(f"  {'点数':>4}{'回収率':>8}{'総収支':>13}{'的中率':>8}{'万舟':>6}")
    best = (0, -1.0)
    for k in GRID:
        ret = sum(r["yen"] for r in rows if r["rk"] <= k)
        man = sum(1 for r in rows if r["rk"] <= k and r["yen"] >= MANSHU)
        stake = n * k * 100
        roi = ret / stake
        pl = ret - stake
        hit = sum(1 for r in rows if r["rk"] <= k) / n
        if roi > best[1]:
            best = (k, roi)
        print(f"  {k:>4}{roi*100:>7.0f}%{pl:>+12,.0f}{hit*100:>7.0f}%{man:>6}")
    peak_pl = max(GRID, key=lambda k: sum(r["yen"] for r in rows if r["rk"] <= k) - n * k * 100)
    pl_at = sum(r["yen"] for r in rows if r["rk"] <= peak_pl) - n * peak_pl * 100
    print(f"\n  ★効率(回収率)ピーク = {best[0]}点（{best[1]*100:.0f}%）")
    print(f"  ★総収支ピーク = {peak_pl}点（{pl_at:+,.0f}円）")
    print("\n※効率(ROI)ピークと総収支ピークは別: 少点数=高ROI・小額 / 多点数=低ROI・大額(レバレッジ)。")
    print("  確定払戻=ライブ未証明。薄プール自己インパクトは多点数(万舟依存)ほど効く＝ライブは効率ピーク寄りが安全。")


if __name__ == "__main__":
    main()
