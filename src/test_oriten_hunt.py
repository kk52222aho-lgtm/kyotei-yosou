"""勝てる目のハンティング: 市場のコース偏重 × lap信号の食い違いを狙い撃つ。

公開信号で唯一効くのは②嗜好の歪み(素人票の1号艇過剰買い)。速いlapの非1号艇が
過小評価される所、遅いlapの1号艇が過大評価される所を、細かい条件で探す。
in-sample探索(多重比較でCI下限>1.0が偶然出るので、survivorは別窓OOSで要確認)。

使い方: python test_oriten_hunt.py --start 20260601 --end 20260619
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
    p = pd.read_sql("SELECT date,jcd,rno,tansho_lane,tansho_yen,"
                    "exacta_combo,exacta_yen,quinella_combo,quinella_yen "
                    "FROM payouts WHERE date BETWEEN ? AND ? AND tansho_yen IS NOT NULL",
                    conn, params=(start, end))
    conn.close()
    df = o.merge(e, on=["date", "jcd", "rno", "lane"])
    df["lap_metric"] = df["lap"].fillna(df["halflap"])
    key = ["date", "jcd", "rno"]
    df["rk"] = df.groupby(key)["lap_metric"].rank(method="first")
    a1 = (df.assign(a=(df["racer_class"] == "A1").astype(int))
            .groupby(key)["a"].sum().rename("a1"))
    df = df.merge(a1, on=key)
    # レース単位の要約(全て group-key index で揃える)
    d2 = df.sort_values(key + ["rk"])
    agg = d2.groupby(key).agg(
        lap1_lane=("lane", "first"),
        a1=("a1", "first"),
        lap1=("lap_metric", lambda s: s.iloc[0]),
        lap2=("lap_metric", lambda s: s.iloc[1] if len(s) > 1 else np.nan),
    )
    agg["lap1_lane"] = agg["lap1_lane"].astype(int)
    agg["gap"] = agg["lap2"] - agg["lap1"]
    lane1 = df[df["lane"] == 1].set_index(key)["rk"].rename("inner_laprk")
    race = agg.join(lane1)
    return df, p, race.reset_index()


def boot_ci(cost, ret, rng, nb=3000):
    n = len(ret)
    roi = ret.sum() / cost.sum() if cost.sum() else np.nan
    boot = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        c = cost[i].sum()
        boot.append(ret[i].sum() / c if c else np.nan)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return roi, lo, hi


def eval_tansho(df, p, mask_race, rng, race):
    """race[mask_race] の対象レースで lap1位艇を単勝。"""
    keys = race[mask_race][["date", "jcd", "rno"]]
    pick = df[df["rk"] == 1].merge(keys, on=["date", "jcd", "rno"])
    m = pick.merge(p, on=["date", "jcd", "rno"])
    if len(m) < 30:
        return None
    cost = np.full(len(m), 100.0)
    ret = np.where(m["lane"] == m["tansho_lane"], m["tansho_yen"], 0.0).astype(float)
    roi, lo, hi = boot_ci(cost, ret, rng)
    return len(m), (ret > 0).mean(), roi, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    df, p, race = load(args.start, args.end)
    rng = np.random.default_rng(0)
    gap_med = race["gap"].median()

    R = race
    strats = {
        "lap1位 全体": np.ones(len(R), bool),
        "lap1位 & 一般(A1<=1)": R["a1"] <= 1,
        "lap1位 & 非1号艇": R["lap1_lane"] != 1,
        "lap1位 & 外(lane>=4)": R["lap1_lane"] >= 4,
        "lap1位 & lane2-3": R["lap1_lane"].isin([2, 3]),
        "1号艇lap遅(rk>=3)→lap1位買": R["inner_laprk"] >= 3,
        "1号艇lap遅(rk>=4)→lap1位買": R["inner_laprk"] >= 4,
        "1号艇lap遅(>=4)&一般→lap1位": (R["inner_laprk"] >= 4) & (R["a1"] <= 1),
        "lap gap大(>中央) 全体": R["gap"] > gap_med,
        "lap gap大 & 非1号艇": (R["gap"] > gap_med) & (R["lap1_lane"] != 1),
        "lap1位 一般&非1号艇": (R["a1"] <= 1) & (R["lap1_lane"] != 1),
        "lap1位 一般&非1&gap大": (R["a1"] <= 1) & (R["lap1_lane"] != 1) & (R["gap"] > gap_med),
    }
    print(f"探索窓 {args.start}-{args.end}  総レース {len(R)}  gap中央={gap_med:.2f}")
    print(f"{'戦略':34s} {'n':>4s} {'的中':>5s} {'ROI':>6s} {'95%CI':>16s}  判定")
    hits = []
    for name, mask in strats.items():
        r = eval_tansho(df, p, np.asarray(mask), rng, race)
        if r is None:
            print(f"  {name:32s}  n不足"); continue
        n, hit, roi, lo, hi = r
        flag = "★下限>1" if lo > 1 else ("×" if hi < 1 else "±")
        if lo > 1:
            hits.append(name)
        print(f"  {name:32s} {n:4d} {hit:5.1%} {roi:6.3f} [{lo:5.3f},{hi:5.3f}]  {flag}")
    print("\n★下限>1(in-sample候補・要OOS確認):", hits if hits else "なし")


if __name__ == "__main__":
    main()
