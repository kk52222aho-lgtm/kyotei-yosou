"""場 × 2連単の目 の出現率と回収率。予測を一切使わん、ただの頻度表。

問い: 「この場で 1-3 が来る率」は? 場別イン勝率と 2連単1-2 の全体ベンチはあったが、
**場 × 特定の目** の表は無かった。

規律:
- 6艇そろい・確定払戻のみ(飛び艇の survivorship 除去)
- **頻度と回収率を必ず並べる。**頻度が高い目は配当が低いので、頻度だけ見ても意味が無い
- 24場 × 30目 = **720セル**の多重比較。上位セルを単体で読まん。
  「偶然でどこまで出るか」を並べ替えで出して、それを超えた分だけ見る
- CI は日付クラスタのブートストラップ

usage:
  python -u -m src.test_combo_by_venue          # 1-3 を主役に全場
  python -u -m src.test_combo_by_venue 1-2      # 目を指定
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import storage
from .venues import name as venue_name

RNG = np.random.default_rng(20260902)
NBOOT = 2000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT date, jcd, rno, exacta_combo, exacta_yen, tansho_lane
        FROM payouts WHERE exacta_combo IS NOT NULL AND exacta_yen IS NOT NULL
    """, conn)
    n6 = pd.read_sql_query("""
        SELECT date, jcd, rno, COUNT(*) n FROM entries GROUP BY date, jcd, rno
    """, conn)
    conn.close()
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    n6["jcd"] = n6["jcd"].astype(str).str.zfill(2)
    df = df.merge(n6, on=["date", "jcd", "rno"], how="inner")
    return df[df["n"] == 6].drop(columns="n").reset_index(drop=True)


def ci(hit, yen, dates):
    """日付クラスタ bootstrap で (的中率CI, 回収率CI)。"""
    ud, iv = np.unique(dates, return_inverse=True)
    h = np.bincount(iv, weights=hit, minlength=len(ud))
    y = np.bincount(iv, weights=yen, minlength=len(ud))
    c = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    cc = c[idx].sum(1)
    return (np.percentile(h[idx].sum(1) / cc, [2.5, 97.5]),
            np.percentile(y[idx].sum(1) / (cc * 100), [2.5, 97.5]))


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    df = load()
    df["hit"] = (df["exacta_combo"] == target).astype(float)
    df["yen"] = np.where(df["hit"] > 0, df["exacta_yen"], 0.0)
    print(f"盤: {len(df):,}レース / {df['date'].nunique():,}日 "
          f"({df['date'].min()}-{df['date'].max()})  6艇そろいのみ\n")

    a, b = df["hit"].mean(), df["yen"].sum() / (len(df) * 100)
    hc, yc = ci(df["hit"].to_numpy(), df["yen"].to_numpy(), df["date"].to_numpy())
    print(f"=== 全場 {target} ===")
    print(f"  出現率 {a:.2%} CI[{hc[0]:.2%},{hc[1]:.2%}]   "
          f"回収率 {b:.1%} CI[{yc[0]:.1%},{yc[1]:.1%}]   "
          f"平均配当 {df.loc[df.hit > 0, 'exacta_yen'].mean():,.0f}円")

    print(f"\n=== 場別 {target}(出現率の高い順) ===")
    print(f"  {'場':<8}{'レース':>8}{'出現率':>8}{'95%CI':>18}{'回収率':>8}"
          f"{'95%CI':>20}{'平均配当':>10}")
    rows = []
    for jcd, g in df.groupby("jcd"):
        h, y = g["hit"].to_numpy(), g["yen"].to_numpy()
        hh, yy = h.mean(), y.sum() / (len(g) * 100)
        hci, yci = ci(h, y, g["date"].to_numpy())
        rows.append({"jcd": jcd, "name": venue_name(jcd), "n": len(g),
                     "rate": hh, "rlo": hci[0], "rhi": hci[1],
                     "roi": yy, "ylo": yci[0], "yhi": yci[1],
                     "pay": g.loc[g.hit > 0, "exacta_yen"].mean()})
    R = pd.DataFrame(rows).sort_values("rate", ascending=False)
    for _, r in R.iterrows():
        print(f"  {r['name']:<8}{r['n']:>8,}{r['rate']:>8.2%}"
              f"  [{r['rlo']:.2%},{r['rhi']:.2%}]{r['roi']:>8.1%}"
              f"  [{r['ylo']:.1%},{r['yhi']:.1%}]{r['pay']:>10,.0f}")

    # ---- 多重比較: 24場 × 30目 = 720セルで、偶然どこまで出るか ----
    print(f"\n=== 720セル(24場 × 30目)の全体像 ===")
    combos = [f"{i}-{j}" for i in range(1, 7) for j in range(1, 7) if i != j]
    cells = []
    for jcd, g in df.groupby("jcd"):
        n = len(g)
        for c in combos:
            m = (g["exacta_combo"] == c)
            cells.append({"jcd": jcd, "name": venue_name(jcd), "combo": c, "n": n,
                          "rate": m.mean(),
                          "roi": g.loc[m, "exacta_yen"].sum() / (n * 100)})
    C = pd.DataFrame(cells)
    print(f"  回収率100%超のセル: {(C['roi'] > 1).sum()} / {len(C)}")
    print(f"  回収率の中央値 {C['roi'].median():.1%} / 最大 {C['roi'].max():.1%}"
          f"(控除25%なので期待は75%前後)")
    print("\n  回収率 上位8セル(※720セル拾いなので、そのまま信じたらあかん):")
    for _, r in C.sort_values("roi", ascending=False).head(8).iterrows():
        print(f"    {r['name']:<8}{r['combo']:<6}出現{r['rate']:>7.2%}"
              f"  回収{r['roi']:>7.1%}  n={r['n']:,}")

    # 帰無: 場をシャッフルして、最大回収率がどこまで行くか
    mx = []
    jc = df["jcd"].to_numpy()
    for _ in range(200):
        sh = RNG.permutation(jc)
        t = df.assign(jcd=sh)
        best = 0.0
        for jcd, g in t.groupby("jcd"):
            n = len(g)
            v = g.groupby("exacta_combo")["exacta_yen"].sum() / (n * 100)
            best = max(best, v.max())
        mx.append(best)
    thr = np.percentile(mx, 95)
    print(f"\n  場をシャッフルした帰無での最大回収率(95%点): {thr:.1%}")
    print(f"  実測の最大 {C['roi'].max():.1%} → "
          f"{'閾値超え' if C['roi'].max() > thr else '**偶然の範囲内**'}")


if __name__ == "__main__":
    main()
