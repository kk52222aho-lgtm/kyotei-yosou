"""事前登録した2戦略のOOS確認(多重比較の蜃気楼を潰す)。

凍結戦略(2026-06探索で中心ROI>1だったもの):
  S1: 1号艇のlap順位>=3 のレースで lap1位艇を単勝
  S2: 一般戦(A1<=1)かつ lap1位艇が非1号艇 のレースで lap1位艇を単勝

--oos で指定した窓(探索と別期間)で ROI+ブートCI を出す。ROI<1に落ちれば棄却。
--pool で探索窓とOOS窓の合算(検出力up、CI下限>1.0が出るか)も見る。

使い方: python test_oriten_oos.py --oos-start 20260401 --oos-end 20260531
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import storage

KEY = ["date", "jcd", "rno"]


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
    if len(o) == 0:
        return None
    df = o.merge(e, on=KEY + ["lane"])
    df["lap_metric"] = df["lap"].fillna(df["halflap"])
    df["rk"] = df.groupby(KEY)["lap_metric"].rank(method="first")
    a1 = (df.assign(a=(df["racer_class"] == "A1").astype(int))
            .groupby(KEY)["a"].sum().rename("a1"))
    df = df.merge(a1, on=KEY)
    inner = df[df["lane"] == 1].set_index(KEY)["rk"].rename("inner_laprk")
    df = df.merge(inner, on=KEY, how="left")
    return df, p


def picks_S1(df):
    r = df[(df["rk"] == 1) & (df["inner_laprk"] >= 3)]
    return r[KEY + ["lane"]]


def picks_S2(df):
    r = df[(df["rk"] == 1) & (df["a1"] <= 1) & (df["lane"] != 1)]
    return r[KEY + ["lane"]]


def picks_S1sharp(df):
    # 構造で研いだ版: 1号艇lap rk∈{3,4} × pick lane∈{2,3,4}
    r = df[(df["rk"] == 1) & (df["inner_laprk"].isin([3, 4])) & (df["lane"].isin([2, 3, 4]))]
    return r[KEY + ["lane"]]


def roi(pick, p, rng, nb=4000):
    m = pick.merge(p, on=KEY)
    if len(m) < 20:
        return None
    ret = np.where(m["lane"] == m["tansho_lane"], m["tansho_yen"], 0.0).astype(float)
    n = len(ret); r = ret.sum() / (100 * n)
    boot = [rng.choice(ret, n, replace=True).sum() / (100 * n) for _ in range(nb)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flag = "★下限>1(エッジ確定的)" if lo > 1 else ("×棄却(市場効率)" if hi < 1 else "±未確定")
    return n, (ret > 0).mean(), r, lo, hi, flag


def report(tag, df, p, rng):
    for name, fn in [("S1: 1号艇lap>=3->lap1位", picks_S1),
                     ("S2: 一般&非1号艇 lap1位", picks_S2),
                     ("S1': 内rk3-4 x pick2-4", picks_S1sharp)]:
        res = roi(fn(df), p, rng)
        if res is None:
            print(f"  [{tag}] {name}: n不足"); continue
        n, hit, r, lo, hi, flag = res
        print(f"  [{tag}] {name:24s} n={n:4d} 的中{hit:.1%} ROI={r:.3f} CI[{lo:.3f},{hi:.3f}] {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", required=True)
    ap.add_argument("--oos-end", required=True)
    ap.add_argument("--disc-start", default="20260601")
    ap.add_argument("--disc-end", default="20260619")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    oos = load(args.oos_start, args.oos_end)
    if oos is None:
        print("OOSデータ未収集(oritenが空)。収集完了を待つこと。"); return
    dfo, po = oos
    print(f"OOS窓 {args.oos_start}-{args.oos_end}: {dfo.groupby(KEY).ngroups}レース")
    report("OOS", dfo, po, rng)

    disc = load(args.disc_start, args.disc_end)
    if disc:
        dfd, pd_ = disc
        dfp = pd.concat([dfd, dfo]); pp = pd.concat([pd_, po])
        print(f"\n合算(探索+OOS): {dfp.groupby(KEY).ngroups}レース")
        report("POOL", dfp, pp, rng)
    print("\n判定: OOSでROI>1維持=survivor / <1=蜃気楼棄却。合算CI下限>1でエッジ確定的。")


if __name__ == "__main__":
    main()
