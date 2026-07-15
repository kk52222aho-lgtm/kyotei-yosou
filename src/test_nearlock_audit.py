"""①near-lockテーブルの監査: 確信→的中(+単勝ROI)を N・bootstrap CI 付きで検品。

near-lockページに無検品でハードした 78/81/84/88% を潔白化する内部監査。
モデル本命(top pick)の確信(正規化勝率)帯別に、的中率(N・CI)と単勝ROI(=③のEV現実)を出す。
walk-forward・SELECTION_YEARS除外・6艇クリーン・確定払戻。年別も。
※これはBT(展示込み特徴量)の物差し=②でライブ確信と分布照合が通るまでページには出さない。

例: python -u -m src.test_nearlock_audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen FROM entries e JOIN payouts p
        ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
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
            if s <= 0 or pd.isna(top["tansho_lane"]):
                continue
            won = int(top["tansho_lane"]) == int(top["lane"])
            rows.append({"yr": y, "conf": float(top["p"]) / s * 100,
                         "won": int(won),
                         "ret": float(top["tansho_yen"]) if won else 0.0})
    return pd.DataFrame(rows)


def ci_hit(w, iters=3000):
    a = np.asarray(w, float); n = len(a)
    rng = np.random.default_rng(9)
    m = [a[rng.integers(0, n, n)].mean() * 100 for _ in range(iters)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    df = load()
    d = collect(df)
    print(f"モデル本命(top pick) 総数 {len(d):,}  対象年 {sorted(d['yr'].unique())}\n")
    print("確信帯(正規化%) :   N     的中%  [CI95]        単勝ROI  ← ③EV現実")
    for lo in [65, 70, 75, 80, 85]:
        sub = d[d["conf"] >= lo]
        n = len(sub)
        if n < 30:
            print(f"  ≥{lo}%: N{n} 少（≥{lo}のサンプル不足=この帯は数字を出せない）")
            continue
        hit = sub["won"].mean() * 100
        lo_ci, hi_ci = ci_hit(sub["won"])
        roi = sub["ret"].sum() / (n * 100) * 100
        yrs = " ".join(f"{y[2:]}:{d[(d.conf>=lo)&(d.yr==y)]['won'].mean()*100:.0f}"
                       for y in sorted(d["yr"].unique()))
        print(f"  ≥{lo}%: {n:6d}  {hit:5.1f}  [{lo_ci:.0f},{hi_ci:.0f}]   {roi:5.1f}%   年{yrs}")
    print("\n判定材料: 各帯の的中CIが締まってるか(N十分か)・年別that安定か・ROIが100%割れか。")
    print("旧ハード値(≥.80=84等)と実測が一致するか。※②(ライブ確信=展示前 と この物差しの分布照合)")
    print("that通るまでページ非公開。的中を出すならROI(このEV現実)も必ず併記。")


if __name__ == "__main__":
    main()
