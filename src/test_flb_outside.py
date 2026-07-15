"""6ptを埋めた候補=「強い非イン本命」の過小人気を、蜃気楼でないか徹底検証する。

test_flb_close: モデル本命that非イン(2-6号)かつ強い(p0≥.55)レースの単勝ROI=110.9%・3年全部
100%超。メカ=イン脳バイアスで客that強い外本命を過小人気にする=クリーンな嗜好の歪み。
だがN=501と小さい→蜃気楼か本物か潰す:
  ①p0閾値スイープ(滑らかに効くか=単発でない)
  ②ファットテール(上位配当を抜いてもROI>100残るか)
  ③bootstrap CI(下限が100超えるか)
  ④年別一貫
全部通れば本物。確定払戻・walk-forward・6艇クリーン。

例: python -u -m src.test_flb_outside
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


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
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
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
            if s <= 0 or int(top["lane"]) == 1:      # 本命=外(2-6号)だけ
                continue
            p0 = float(top["p"]) / s
            won = int(top["tansho_lane"]) == int(top["lane"])
            rows.append({"yr": y, "p0": p0, "won": int(won),
                         "ret": float(top["tansho_yen"]) if won else 0.0})
    return pd.DataFrame(rows)


def boot_ci(ret, iters=3000):
    arr = ret.to_numpy()
    n = len(arr)
    rng = np.random.default_rng(42)
    means = [arr[rng.integers(0, n, n)].sum() / (n * 100) * 100 for _ in range(iters)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    df = load()
    d = collect(df)
    print(f"本命=外(2-6号) 総数 {len(d):,}\n")

    print("[① p0閾値スイープ: 滑らかに効けば本物・単発なら怪しい]")
    print("  閾値      N    的中   ROI    ROI(上位3配当抜)  bootCI95%      年別")
    for lo in [0.40, 0.45, 0.50, 0.55, 0.60]:
        sub = d[d["p0"] >= lo]
        n = len(sub)
        if n < 50:
            print(f"  p0≥{lo:.2f}: n{n} 少")
            continue
        roi = sub["ret"].sum() / (n * 100) * 100
        # ②ファットテール: 上位3配当を除いたROI
        cut = sub["ret"].nlargest(3).sum()
        roi_trim = (sub["ret"].sum() - cut) / (n * 100) * 100
        lo_ci, hi_ci = boot_ci(sub["ret"])
        hr = sub["won"].mean() * 100
        yrs = " ".join(f"{y}:{sub[sub.yr==y]['ret'].sum()/(len(sub[sub.yr==y])*100)*100:.0f}"
                       for y in sorted(d["yr"].unique()))
        star = " ★" if lo_ci >= 100 else ("" if roi >= 100 else "")
        print(f"  p0≥{lo:.2f}: {n:5d}  {hr:4.1f}%  {roi:5.1f}%   {roi_trim:5.1f}%      "
              f"[{lo_ci:.0f},{hi_ci:.0f}]  {yrs}{star}")

    print("\n判定: p0上げるほどROI↑で滑らか&bootCI下限≥100&上位抜いても>100&年別一貫 → 本物の妙味。")
    print("      ROIが上位配当依存(trimで急落)orCI下限<100 → まだ蜃気楼。確定払戻・6艇クリーン。")


if __name__ == "__main__":
    main()
