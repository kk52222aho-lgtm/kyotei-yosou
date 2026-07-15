"""本丸: イン過大人気/外過小人気を全艇で叩く=同じ確率でも外艇がインより高く戻るか。

逆イン本命(モデル1位が外)はN597で詰まった=絞りすぎ。歪みの本体は「イン=過大人気・
外=過小人気」。∴全レースの全艇をmodel確率帯で割り、lane==1(イン) vs lane!=1(外)で
ROI/的中/bootCIを比較。同確率で外>インなら市場がインを過大評価=クリーンな歪み。
外艇that100%超える確率帯をN数十万で特定。ここでCI下限≥100が出れば前向き待たず確定。
確定払戻・walk-forward・6艇クリーン。

例: python -u -m src.test_inside_overbet
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
            s = g["p"].sum()
            if s <= 0:
                continue
            win_lane = int(g.iloc[0]["tansho_lane"])
            yen = float(g.iloc[0]["tansho_yen"])
            for r in g.itertuples():
                won = int(r.lane) == win_lane
                rows.append({"yr": y, "mp": float(r.p) / s, "inside": int(int(r.lane) == 1),
                             "won": int(won), "ret": yen if won else 0.0})
    return pd.DataFrame(rows)


def ci(ret, iters=3000):
    a = ret.to_numpy(); n = len(a)
    rng = np.random.default_rng(3)
    m = [a[rng.integers(0, n, n)].sum() / (n * 100) * 100 for _ in range(iters)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def cell(sub):
    n = len(sub)
    if n < 200:
        return f"n{n}少"
    roi = sub["ret"].sum() / (n * 100) * 100
    lo, hi = ci(sub["ret"])
    star = "★" if lo >= 100 else ("○" if roi >= 100 else "")
    return f"{roi:5.1f}% [{lo:.0f},{hi:.0f}] n{n}{star}"


def main():
    df = load()
    d = collect(df)
    print(f"全艇 総数 {len(d):,}\n")
    edges = [0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, 1.01]
    d["bkt"] = pd.cut(d["mp"], edges, right=False)
    print("model確率帯   :   イン(1号)ROI          外(2-6号)ROI          差(外-イン)")
    for b, sub in d.groupby("bkt", observed=True):
        ins = sub[sub["inside"] == 1]
        out = sub[sub["inside"] == 0]
        if len(out) < 200:
            continue
        ir = ins["ret"].sum() / (len(ins) * 100) * 100 if len(ins) >= 200 else float("nan")
        orr = out["ret"].sum() / (len(out) * 100) * 100
        diff = orr - ir if not np.isnan(ir) else float("nan")
        print(f"  [{b.left:.2f},{b.right:.2f}): イン {ir:5.1f}%   外 {cell(out):32s}  差{diff:+5.1f}pt")

    print("\n[外艇だけ 確率下限スイープ(★=CI下限≥100=前向き待たず確定)]")
    for lo in [.30, .35, .40, .45, .50, .55]:
        sub = d[(d["inside"] == 0) & (d["mp"] >= lo)]
        print(f"  外×mp≥{lo:.2f}: {cell(sub)}")

    print("\n判定: 同確率帯で外>イン(差+)が一貫=市場がインを過大評価の実在歪み。")
    print("      外艇のどこかでCI下限≥100(★)なら、N厚めで歴史確定=前向き不要。確定払戻・6艇クリーン。")


if __name__ == "__main__":
    main()
