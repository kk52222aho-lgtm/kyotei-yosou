"""強い非イン本命の妙味を、取っといた2016-2022(完全ホールドアウト)でOOS決着させる。

test_flb_outside: 本命=外×p0≥.55で単勝110.9%・勾配滑らか・3年一貫だがN501でbootCI下限97・
上位3配当抜きで99.4=確定に薄い。この仮説はSELECTION_YEARS(2016-22)から作ってない→そこで
OOSテストすればN2-3倍の真の場外検証。各年その前年までで学習(walk-forward)。
勾配/trim/bootCI/年別を全期間で。CI下限that100超えれば本物確定・跨げば「有望だが未確定」。
確定払戻・6艇クリーン。

例: python -u -m src.test_flb_outside_oos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df):
    rows = []
    for y in sorted(df["yr"].unique()):                 # 全年(SELECTION_YEARS含む)をOOSで
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            s = g["p"].sum()
            if s <= 0 or int(top["lane"]) == 1:
                continue
            p0 = float(top["p"]) / s
            won = int(top["tansho_lane"]) == int(top["lane"])
            rows.append({"yr": y, "p0": p0, "won": int(won),
                         "ret": float(top["tansho_yen"]) if won else 0.0})
    return pd.DataFrame(rows)


def boot_ci(ret, iters=4000):
    arr = ret.to_numpy(); n = len(arr)
    rng = np.random.default_rng(7)
    means = [arr[rng.integers(0, n, n)].sum() / (n * 100) * 100 for _ in range(iters)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    df = load()
    d = collect(df)
    yrs = sorted(d["yr"].unique())
    print(f"本命=外(2-6号) 全年OOS 総数 {len(d):,}  対象年 {yrs[0]}-{yrs[-1]}\n")
    print("  閾値      N    的中   ROI    trim上位3   bootCI95%      年別")
    for lo in [0.45, 0.50, 0.55, 0.60]:
        sub = d[d["p0"] >= lo]
        n = len(sub)
        if n < 50:
            print(f"  p0≥{lo:.2f}: n{n} 少"); continue
        roi = sub["ret"].sum() / (n * 100) * 100
        roi_trim = (sub["ret"].sum() - sub["ret"].nlargest(3).sum()) / (n * 100) * 100
        lo_ci, hi_ci = boot_ci(sub["ret"])
        hr = sub["won"].mean() * 100
        yv = " ".join(f"{y[2:]}:{sub[sub.yr==y]['ret'].sum()/(max(len(sub[sub.yr==y]),1)*100)*100:.0f}"
                      for y in yrs if len(sub[sub.yr == y]) > 30)
        star = " ★確定(CI下限≥100)" if lo_ci >= 100 else (" ○点推定超" if roi >= 100 else "")
        print(f"  p0≥{lo:.2f}: {n:5d}  {hr:4.1f}%  {roi:5.1f}%   {roi_trim:5.1f}%   [{lo_ci:.0f},{hi_ci:.0f}]  {yv}{star}")
    print("\n判定: bootCI下限≥100&trim>100&年別大半100超 → 本物のエッジ確定。")
    print("      CI跨ぐ/trim<100 → 有望だが未確定=前向き実測で決着(実弾papertrade)。")


if __name__ == "__main__":
    main()
