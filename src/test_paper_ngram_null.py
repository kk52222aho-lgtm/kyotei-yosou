"""v2 の t=+4.07 を、**学習した列の正しい帰無分布**に当て直す。

## なんで要るか

v2 は FWER 5% 閾値 2.67 を、7列の multiplier bootstrap で出した。
**あの bootstrap は列が固定されとる前提**や。ところが `ngram学習スコア` は
訓練残差に 7,190 個のパラメータを当てて作った列で、**固定した列とは
帰無分布がちがう**。

実際、偽薬(ラベル入替で学習)は holdout で **+1.43 / +2.15** まで出た。
純ノイズ列は +1.23 / +0.30 やったから、**学習を挟むと裾が伸びとる**。
偽薬2本では裾が測れん。

さらに訓練内で年をまたがせたら **2023→2024 t=−0.30 / 2024→2023 t=+0.26** で
転移がゼロやった。学習量が倍で t が 0.26 → 4.07 に跳ぶんは算数が合わん
(√2 倍なら 0.37)。

## やること

**ラベルを壊して学習させる**のを N 回やって、その holdout t の分布を作る。
これが「学習を挟んだ列」の正しい帰無分布や。+4.07 をそこに当てる。

- 入替は**訓練レースの中だけ**(holdout の残差を訓練ラベルに混ぜん)
- alpha は本物が選んだ 1000 に固定(N回 CV を回すのは高いんで)。
  **本物も alpha=1000 やったから、同じ条件で比べとる**
- これは新しい仮説の検定やない。**既にやった検定の帰無分布を測り直しとるだけ**
"""
from __future__ import annotations

import argparse
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .scan_all import load, race_z
from .test_paper_comment import JCD, orthogonalized_resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cutoff", default="2025")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1000.0)
    a = ap.parse_args()

    df = load()
    df = df[df["jcd"] == JCD].copy()
    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    df = df.merge(oof, on=["date", "jcd", "rno", "lane"], how="left")
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    raw = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, comment FROM paper_comment WHERE jcd=?",
        con, params=(JCD,))
    con.close()
    raw["jcd"] = raw["jcd"].astype(str).str.zfill(2)
    raw["rno"] = raw["rno"].astype(int)
    raw["lane"] = raw["lane"].astype(int)
    df = df.merge(raw, on=["date", "jcd", "rno", "lane"], how="left")
    df = df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)

    rid, uniq = pd.factorize(df["rid"], sort=False)
    nrace = len(uniq)
    q, resid, ro, keep, okz, r2 = orthogonalized_resid(df, nrace, rid)
    text = df["comment"].fillna("").map(lambda t: unicodedata.normalize("NFKC", t))
    yr = df["date"].str[:4].to_numpy()
    has = (text.str.len() > 0).to_numpy()
    fit = (yr < a.cutoff) & okz & has & ~np.isnan(ro)
    ho = (yr >= a.cutoff) & okz
    ho = (np.bincount(rid, weights=ho.astype(float), minlength=nrace)[rid] == 6)

    dcodes, dun = pd.factorize(df["dnum"], sort=True)
    nd = len(dun)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          min_df=20, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform(text[fit])
    Xall = vec.transform(text)
    y = np.nan_to_num(ro, nan=0.0)

    def hold_t(pred: np.ndarray) -> float:
        z = race_z(np.where(has, pred, np.nan), nrace)
        m = ho & ~np.isnan(z)
        m = (np.bincount(rid, weights=m.astype(float), minlength=nrace)[rid] == 6)
        W = np.bincount(dcodes, weights=np.where(m, z, 0.0) * np.where(m, ro, 0.0),
                        minlength=nd)
        return float(W.sum() / np.sqrt((W ** 2).sum()))

    real = Ridge(alpha=a.alpha).fit(Xtr, y[fit]).predict(Xall)
    t_real = hold_t(real)
    print(f"本物の holdout t = {t_real:+.2f}(v2 の +4.07 と同じ手。alpha={a.alpha:.0f}固定)")

    tr_race = (np.bincount(rid, weights=(yr < a.cutoff).astype(float),
                           minlength=nrace) == 6)
    tr_idx = np.flatnonzero(tr_race)
    rng = np.random.default_rng(20260924)
    ts = []
    for i in range(a.n):
        Y = y.reshape(nrace, 6).copy()
        Y[tr_idx] = Y[rng.permutation(tr_idx)]
        yb = Y.ravel()
        p = Ridge(alpha=a.alpha).fit(Xtr, yb[fit]).predict(Xall)
        ts.append(hold_t(p))
        print(f"  [{i+1}/{a.n}] ラベル入替で学習 → holdout t = {ts[-1]:+.2f}", flush=True)
    ts = np.array(ts)
    print(f"\n=== 学習を挟んだ列の帰無分布(N={a.n}) ===")
    print(f"  平均 {ts.mean():+.2f} / sd {ts.std(ddof=1):.2f} / "
          f"最大|t| {np.abs(ts).max():.2f}")
    print(f"  |t| の 95%点 ≈ {np.percentile(np.abs(ts), 95):.2f}"
          f"  (v2 が使うた固定列の閾値は 2.67)")
    p = float((np.abs(ts) >= abs(t_real)).mean())
    print(f"  本物 |t|={abs(t_real):.2f} 以上が出た偽薬: "
          f"{int((np.abs(ts) >= abs(t_real)).sum())}/{a.n}  → 経験 p ≈ {p:.3f}")
    print("\n  ※ N が小さいんで p の分解能は 1/N まで。"
          "N=20 やと p<0.05 は『0本』でしか言えん")


if __name__ == "__main__":
    main()
