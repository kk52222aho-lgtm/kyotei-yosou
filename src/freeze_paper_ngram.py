"""前向きに撃つための n-gram モデルを凍らせる。

holdout は2回見た(辞書版 / 学習版)。**3回目はやらん**と書いた。
残っとるのは前向きだけや。前向きにやるには、**撃つ前にモデルを固定して
SHA を刻む**必要がある。→ `docs/prereg_paper_ngram.md`

## 学習に使う範囲

**手元にある全部**(2023-01 〜 凍結日)。holdout(2025-26)も込みや。
前向きの標本は**まだ存在せんレース**やから、過去を全部使うても先読みにならん。
むしろ **13.7k艇では出ず 27.4k で出た**(2026-09-24 の検算)んやから、
本数は多いほどええ。

## 凍らせるもん

    data/paper_ngram.joblib   TF-IDF(char 2-4gram) + Ridge(alpha=1000)
    SHA256 を docs/prereg_paper_ngram.md に書く。**以後いっさい作り直さん。**
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .scan_all import load
from .test_paper_comment import JCD, orthogonalized_resid

ALPHA = 1000.0        # v2 が訓練内CVで選んだ値。振り直さん
OUT = "data/paper_ngram.joblib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--out", default=OUT)
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
    has = (text.str.len() > 0).to_numpy()
    fit = okz & has & ~np.isnan(ro)
    print(f"学習に使う艇 {int(fit.sum()):,} "
          f"({df.loc[fit, 'date'].min()} 〜 {df.loc[fit, 'date'].max()})")

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          min_df=20, max_features=30000, sublinear_tf=True)
    X = vec.fit_transform(text[fit])
    m = Ridge(alpha=ALPHA).fit(X, np.nan_to_num(ro, nan=0.0)[fit])
    print(f"n-gram {X.shape[1]:,} 本 / alpha={ALPHA}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vec": vec, "model": m, "alpha": ALPHA,
                 "trained_rows": int(fit.sum()),
                 "date_min": str(df.loc[fit, "date"].min()),
                 "date_max": str(df.loc[fit, "date"].max())}, a.out)
    sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(f"\n凍結: {a.out}")
    print(f"SHA256 = {sha}")
    print("🚨 この SHA を docs/prereg_paper_ngram.md に写す。以後いっさい作り直さん。")


if __name__ == "__main__":
    main()
