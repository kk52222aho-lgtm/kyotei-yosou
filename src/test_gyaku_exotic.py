"""逆イン本命の歪みをexoticで増幅できるか=強い外本命からの2連単/2連複that単勝より濃いか。

単勝の逆イン本命は110.9%(N501,CI[97,130])で確定手前。客が「外が勝つ」を過小評価してるなら、
その外本命を軸にした2連単/2連複はもっと甘いはず(歪みの増幅)。単勝より高ROIかつCIが締まれば
同N500でも歴史確定に近づく。外本命(model rank1が非イン)×p0閾で:
  単勝 / 2連単(外1着-流しtop2,top3) / 2連複(外絡み2点) のROI・bootCI・年別。
確定払戻・walk-forward・6艇クリーン。

例: python -u -m src.test_gyaku_exotic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen, p.exacta_combo, p.exacta_yen,
               p.quinella_combo, p.quinella_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.exacta_yen IS NOT NULL
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
            if len(g) != 6 or pd.isna(g.iloc[0]["exacta_combo"]):
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            h = int(top["lane"])
            if h == 1:                                  # 外本命だけ
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p0 = float(top["p"]) / s
            others = [int(l) for l in g["lane"] if int(l) != h]     # model確率降順
            ex2 = {f"{h}-{x}" for x in others[:2]}
            ex3 = {f"{h}-{x}" for x in others[:3]}
            qn2 = {"=".join(sorted([str(h), str(x)], key=int)) for x in others[:2]}
            excombo = str(top["exacta_combo"])
            qncombo = (str(top["quinella_combo"]) if not pd.isna(top.get("quinella_combo")) else None)
            won_t = (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == h
            rows.append({
                "yr": y, "p0": p0,
                "t_ret": float(top["tansho_yen"]) if won_t else 0.0,
                "e2_ret": float(top["exacta_yen"]) if excombo in ex2 else 0.0,
                "e3_ret": float(top["exacta_yen"]) if excombo in ex3 else 0.0,
                "q2_ret": (float(top["quinella_yen"]) if (qncombo and qncombo in qn2
                           and not pd.isna(top["quinella_yen"])) else 0.0),
                "q_ok": not pd.isna(top.get("quinella_yen")),
            })
    return pd.DataFrame(rows)


def ci(ret, pts, iters=3000):
    a = ret.to_numpy(); n = len(a)
    rng = np.random.default_rng(5)
    m = [a[rng.integers(0, n, n)].sum() / (n * pts * 100) * 100 for _ in range(iters)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def line(sub, col, pts, label, ok=None):
    d = sub if ok is None else sub[sub[ok]]
    n = len(d)
    if n < 150:
        print(f"    {label:16s}: n{n}少"); return
    roi = d[col].sum() / (n * pts * 100) * 100
    lo, hi = ci(d[col], pts)
    yv = " ".join(f"{y[2:]}:{d[d.yr==y][col].sum()/(max(len(d[d.yr==y]),1)*pts*100)*100:.0f}"
                  for y in sorted(sub["yr"].unique()) if len(d[d.yr == y]) > 20)
    star = "★" if lo >= 100 else ("○" if roi >= 100 else "")
    print(f"    {label:16s}: ROI{roi:5.1f}% [{lo:.0f},{hi:.0f}] n{n} {star}  [{yv}]")


def main():
    df = load()
    d = collect(df)
    print(f"外本命(model rank1が非イン) 総数 {len(d):,}\n")
    for lo in [0.45, 0.50, 0.55]:
        sub = d[d["p0"] >= lo]
        print(f"[外本命 × p0≥{lo:.2f}  (n{len(sub)})]")
        line(sub, "t_ret", 1, "単勝1点")
        line(sub, "e2_ret", 2, "2連単 外1着-2点")
        line(sub, "e3_ret", 3, "2連単 外1着-3点")
        line(sub, "q2_ret", 2, "2連複 外絡2点", ok="q_ok")
        print()
    print("判定: exoticのどれかthat単勝(110%)より高ROI&CI下限≥100(★) → 歪み増幅で歴史確定。")
    print("      全部単勝以下 → 増幅せず=単勝の逆イン本命that本体、前向き実測へ。確定払戻・6艇クリーン。")


if __name__ == "__main__":
    main()
