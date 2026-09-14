"""A1濃度(レース内A1人数)を無料のグレード代理に、oritenのエッジ/予測力を層別検定。

仮説: 大レース(A1集中)は売上でかく市場効率的で勝てない。薄い一般戦(A1少)は
素人票の歪みが残る=単勝エッジ/展示信頼性がそこに宿るかもしれない。プールでは均される。

- ①単勝EV: lap1位艇の単勝ROI+レース単位ブートCI をA1濃度ビン別に。
- ②予測力: lap1位艇の1着率 と Spearman(lap_rank vs 着)をビン別に(展示の効き方の差)。

使い方: python test_oriten_segment.py --start 20260601 --end 20260619
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import storage


def load(start, end):
    conn = sqlite3.connect(storage.DB_PATH)
    o = pd.read_sql("SELECT date,jcd,rno,lane,lap,halflap FROM oriten "
                    "WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    e = pd.read_sql("SELECT date,jcd,rno,lane,finish,racer_class FROM entries "
                    "WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    p = pd.read_sql("SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
                    "WHERE date BETWEEN ? AND ? AND tansho_yen IS NOT NULL",
                    conn, params=(start, end))
    conn.close()
    df = o.merge(e, on=["date", "jcd", "rno", "lane"])
    df["lap_metric"] = df["lap"].fillna(df["halflap"])
    key = ["date", "jcd", "rno"]
    df["rk"] = df.groupby(key)["lap_metric"].rank(method="first")
    # レース内A1人数
    a1 = (df.assign(isA1=(df["racer_class"] == "A1").astype(int))
            .groupby(key)["isA1"].sum().rename("a1cnt"))
    df = df.merge(a1, on=key)
    return df, p


def bin_label(n):
    if n <= 1:
        return "A1_0-1 (薄い/一般寄り)"
    if n <= 3:
        return "A1_2-3 (中間)"
    return "A1_4-6 (集中/グレード寄り)"


def roi_ci(pick, p, rng):
    m = pick.merge(p, on=["date", "jcd", "rno"])
    if len(m) < 20:
        return None
    ret = np.where(m["pl"] == m["tansho_lane"], m["tansho_yen"], 0.0).astype(float)
    n = len(ret); roi = ret.sum() / (100 * n)
    boot = [rng.choice(ret, n, replace=True).sum() / (100 * n) for _ in range(3000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flag = "＋エッジ疑" if lo > 1 else ("×市場効率" if hi < 1 else "±差なし")
    return n, (ret > 0).mean(), roi, lo, hi, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    df, p = load(args.start, args.end)
    df["seg"] = df["a1cnt"].map(bin_label)
    rng = np.random.default_rng(0)

    order = ["A1_0-1 (薄い/一般寄り)", "A1_2-3 (中間)", "A1_4-6 (集中/グレード寄り)"]
    print("=== レース分布(A1濃度) ===")
    rc = df.drop_duplicates(["date", "jcd", "rno"])["seg"].value_counts()
    for s in order:
        print(f"  {s}: {rc.get(s,0)}レース")

    print("\n=== ①単勝EV: lap1位艇のROI (ビン別) ===")
    for s in order:
        sub = df[df["seg"] == s]
        pick = sub[sub["rk"] == 1][["date", "jcd", "rno", "lane"]].rename(columns={"lane": "pl"})
        r = roi_ci(pick, p, rng)
        if r is None:
            print(f"  [{s}] n不足"); continue
        n, hit, roi, lo, hi, flag = r
        print(f"  [{s:24s}] n={n:4d} 的中{hit:.1%} ROI={roi:.3f} CI[{lo:.3f},{hi:.3f}] {flag}")
    # 参照: 全体
    pick = df[df["rk"] == 1][["date", "jcd", "rno", "lane"]].rename(columns={"lane": "pl"})
    n, hit, roi, lo, hi, flag = roi_ci(pick, p, rng)
    print(f"  [{'全体(プール)':24s}] n={n:4d} 的中{hit:.1%} ROI={roi:.3f} CI[{lo:.3f},{hi:.3f}] {flag}")

    print("\n=== ②予測力: lap1位艇の1着率 と Spearman (ビン別) ===")
    for s in order:
        sub = df[df["seg"] == s].copy()
        lap1 = sub[sub["rk"] == 1]
        wr = (lap1["finish"] == 1).mean()
        base = (sub["finish"] == 1).mean()  # =1/艇数
        rho = sub["rk"].corr(sub["finish"], method="spearman")
        nr = sub.drop_duplicates(["date", "jcd", "rno"]).shape[0]
        print(f"  [{s:24s}] {nr}R lap1着率={wr:.3f}(ベース{base:.3f}) 上振れ{wr-base:+.3f} Spearman={rho:+.3f}")


if __name__ == "__main__":
    main()
