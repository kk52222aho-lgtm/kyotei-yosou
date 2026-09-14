"""オリジナル展示(一周タイム)順位から組むエグゾチック(2連単/3連単等)のEV検定。

単勝は市場効率で織込済(ROI CI上限<1.0 or 1.0跨ぎ)。エグゾチック市場は
コース偏重・本命-穴バイアス(②嗜好の歪み)が濃く、公開信号でも歪みが残りうる。
lap順位ベースの各種買い目を実配当(payouts)に対しレース単位ブートCIで判定する。

買い目コスト込みでROI=Σ配当/Σ投資。CI下限>1.0で初めて回収エッジ疑い。

使い方: python test_oriten_exotic.py --start 20260601 --end 20260619
"""
from __future__ import annotations

import argparse
import sqlite3
from itertools import permutations

import numpy as np
import pandas as pd

import storage


def load(start, end):
    conn = sqlite3.connect(storage.DB_PATH)
    o = pd.read_sql(
        "SELECT date,jcd,rno,lane,lap,halflap FROM oriten "
        "WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    p = pd.read_sql(
        "SELECT date,jcd,rno,exacta_combo,exacta_yen,trifecta_combo,trifecta_yen,"
        "trio_combo,trio_yen,quinella_combo,quinella_yen FROM payouts "
        "WHERE date BETWEEN ? AND ? AND trifecta_yen IS NOT NULL",
        conn, params=(start, end))
    conn.close()
    o["lap_metric"] = o["lap"].fillna(o["halflap"])
    o = o.dropna(subset=["lap_metric"])
    o["rk"] = o.groupby(["date", "jcd", "rno"])["lap_metric"].rank(method="first")
    return o, p


def lap_order(g):
    """レースの lap 順位 → [lane(rk1), lane(rk2), ...]"""
    s = g.sort_values("rk")
    return list(s["lane"].astype(int))


def build_races(o, p):
    """レースごとに {lap順の艇番リスト, 各配当combo/yen} をまとめる。"""
    races = {}
    for key, g in o.groupby(["date", "jcd", "rno"]):
        order = lap_order(g)
        if len(order) < 3:
            continue
        races[key] = {"order": order}
    pp = p.set_index(["date", "jcd", "rno"])
    rows = []
    for key, d in races.items():
        if key not in pp.index:
            continue
        r = pp.loc[key]
        rows.append({
            "order": d["order"],
            "exacta_combo": r["exacta_combo"], "exacta_yen": r["exacta_yen"],
            "trifecta_combo": r["trifecta_combo"], "trifecta_yen": r["trifecta_yen"],
            "trio_combo": r["trio_combo"], "trio_yen": r["trio_yen"],
            "quinella_combo": r["quinella_combo"], "quinella_yen": r["quinella_yen"],
        })
    return rows


# --- 買い目生成: (bet_type, [combo文字列...]) を返す ---
def s_exacta_12(o):        # 2連単 lap1→lap2
    return "exacta", [f"{o[0]}-{o[1]}"]
def s_exacta_1x23(o):      # 2連単 lap1固定→lap2,3 (2点)
    return "exacta", [f"{o[0]}-{o[1]}", f"{o[0]}-{o[2]}"]
def s_quinella_12(o):      # 2連複 {lap1,lap2}
    a, b = sorted([o[0], o[1]])
    return "quinella", [f"{a}-{b}"]
def s_trio_123(o):         # 3連複 {lap1,2,3}
    a, b, c = sorted([o[0], o[1], o[2]])
    return "trio", [f"{a}-{b}-{c}"]
def s_trifecta_123(o):     # 3連単 lap1-2-3 (1点)
    return "trifecta", [f"{o[0]}-{o[1]}-{o[2]}"]
def s_trifecta_1_23_23(o): # 3連単 lap1固定1着→2,3着=lap2,3,4の並べ(6点)
    tops = o[1:4]
    combos = [f"{o[0]}-{a}-{b}" for a, b in permutations(tops, 2)]
    return "trifecta", combos

STRATS = {
    "2連単 lap1→2 (1点)": s_exacta_12,
    "2連単 lap1→2,3 (2点)": s_exacta_1x23,
    "2連複 {lap1,2} (1点)": s_quinella_12,
    "3連複 {lap1,2,3} (1点)": s_trio_123,
    "3連単 lap1-2-3 (1点)": s_trifecta_123,
    "3連単 lap1軸→2,3着4艇 (6点)": s_trifecta_1_23_23,
}

COMBO_COL = {"exacta": ("exacta_combo", "exacta_yen"),
             "quinella": ("quinella_combo", "quinella_yen"),
             "trio": ("trio_combo", "trio_yen"),
             "trifecta": ("trifecta_combo", "trifecta_yen")}


def eval_strat(rows, fn, n_boot=3000, seed=0):
    cost = np.zeros(len(rows))
    ret = np.zeros(len(rows))
    for i, r in enumerate(rows):
        btype, combos = fn(r["order"])
        ccol, ycol = COMBO_COL[btype]
        cost[i] = 100 * len(combos)
        win_combo = r[ccol]
        if win_combo in combos and pd.notna(r[ycol]):
            ret[i] = float(r[ycol])
    roi = ret.sum() / cost.sum()
    hit = (ret > 0).mean()
    rng = np.random.default_rng(seed)
    n = len(rows)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot.append(ret[idx].sum() / cost[idx].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flag = "＋エッジ疑" if lo > 1.0 else ("×市場効率" if hi < 1.0 else "±有意差なし")
    return roi, lo, hi, hit, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    o, p = load(args.start, args.end)
    rows = build_races(o, p)
    print(f"検定レース数: {len(rows)} (lap順位×実配当が揃うレース)")
    print(f"{'戦略':32s} {'ROI':>6s} {'95%CI':>18s} {'的中':>6s}  判定")
    for name, fn in STRATS.items():
        roi, lo, hi, hit, flag = eval_strat(rows, fn)
        print(f"  {name:30s} {roi:6.3f} [{lo:5.3f},{hi:5.3f}] {hit:5.1%}  {flag}")
    print("  判定: CI下限>1.0=回収エッジ疑い / CI上限<1.0=市場効率 / 跨ぎ=有意差なし")


if __name__ == "__main__":
    main()
