"""n-gram の信号は「中身」か「言い回し(書き手の癖)」か。

`docs/design_paper_comment_v2.md` の結果に書いた**次の検算**をそのままやる。
**holdout は一切見ん。訓練期間(2023-2024)の中だけ**で回す。

## なんでやるか

holdout を通った学習スコアの「来ん側」の上位に
**「ので」「する」「って」「たけど」**いう機能語が並んどった。筋は2つ:

1. 言い回しが評価と結びついとる(「〜ので、調整する」= 不足の説明)
2. **書き手/時期の癖を拾っとる**(誰の談話か、いつの紙かの代理)

2やと、盤の外の情報やのうて**選手の同定を学習し直しとる**だけかもしれん。

## 割り方(char n-gram に内容語/機能語の区別は無いんで、字種で代理する)

    内容語だけ … 漢字かカタカナを含む n-gram だけ残す
    機能語だけ … ひらがな(と記号)だけの n-gram だけ残す

完全な切り分けやない。**代理やと分かった上で使う。**

## 読み方(回す前に決める)

| 機能語だけの版 | 読み方 |
|---|---|
| 訓練内CV相関がほぼ0 | 信号は**中身**に在る。疑いは薄まる |
| 全部版に近い | 信号は**言い回し**に在る。書き手/時期の癖の疑いが濃い |
| 中間 | どっちもある。**断定せん** |

おまけの点呼: 学習スコアが `nat_win` のレース内順位とどれだけ相関しとるか。
高かったら「選手の強さを学習し直しとる」いう直接の証拠になる。
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .scan_all import load, race_z
from .test_paper_comment import JCD, orthogonalized_resid

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
KANJI_KATA = re.compile(r"[一-龥ァ-ヶー]")


def cv_corr(X, y, groups, alphas=ALPHAS) -> tuple[float, float]:
    """訓練内 GroupKFold(日) で alpha を選び、その CV 相関を返す。"""
    gkf = GroupKFold(n_splits=4)
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        sc = []
        for tri, tei in gkf.split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tri], y[tri])
            p = m.predict(X[tei])
            sc.append(0.0 if p.std() == 0 else float(np.corrcoef(p, y[tei])[0, 1]))
        s = float(np.nanmean(sc))
        if s > best:
            best, best_a = s, a
    return best, best_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cutoff", default="2025")
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

    rid_codes, uniq = pd.factorize(df["rid"], sort=False)
    nrace = len(uniq)
    q, resid, ro, keep, okz, r2 = orthogonalized_resid(df, nrace, rid_codes)

    text = df["comment"].fillna("").map(lambda t: unicodedata.normalize("NFKC", t))
    yr = df["date"].str[:4].to_numpy()
    has = (text.str.len() > 0).to_numpy()
    fit = (yr < a.cutoff) & okz & has & ~np.isnan(ro)
    print(f"🚨 holdout は一切見ん。訓練期間だけ {int(fit.sum()):,} 艇")

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          min_df=20, max_features=30000, sublinear_tf=True)
    X = vec.fit_transform(text[fit])
    names = np.array(vec.get_feature_names_out())
    is_content = np.array([bool(KANJI_KATA.search(s)) for s in names])
    y = np.nan_to_num(ro, nan=0.0)[fit]
    g = df["dnum"].to_numpy()[fit]
    print(f"n-gram {X.shape[1]:,} 本 / 内容語(漢字カナ入り) {int(is_content.sum()):,} "
          f"/ 機能語(ひらがな等のみ) {int((~is_content).sum()):,}")

    print("\n=== 訓練内CV相関(直交化残差に当てる) ===")
    out = {}
    for lab, cols in (("全部", None),
                      ("内容語だけ(ひらがなのみを落とす)", is_content),
                      ("機能語だけ(漢字カナ入りを落とす)", ~is_content)):
        Xs = X if cols is None else X[:, cols]
        c, al = cv_corr(Xs, y, g)
        out[lab] = c
        print(f"  {lab:34s} 次元{Xs.shape[1]:6,}  CV相関 {c:+.4f}  (alpha={al})")

    base = out["全部"]
    fw = out["機能語だけ(漢字カナ入りを落とす)"]
    cw = out["内容語だけ(ひらがなのみを落とす)"]
    print(f"\n  機能語だけ / 全部 = {fw/base:.0%}   内容語だけ / 全部 = {cw/base:.0%}")

    print("\n=== おまけの点呼: 学習スコアは nat_win の焼き直しか ===")
    m = Ridge(alpha=1000.0).fit(X, y)
    pred_all = np.full(len(df), np.nan)
    pred_all[fit] = m.predict(X)
    z_pred = race_z(np.where(has, pred_all, np.nan), nrace)
    z_nat = race_z(df["natwin"].to_numpy(float), nrace)
    ok = ~np.isnan(z_pred) & ~np.isnan(z_nat)
    print(f"  学習スコアのレース内順位 vs nat_win のレース内順位: "
          f"r={np.corrcoef(z_pred[ok], z_nat[ok])[0,1]:+.4f}  (n={int(ok.sum()):,})")
    # 生のコメントから nat_win を当てられるか(当てられたら選手の同定が効いとる)
    ynat = np.nan_to_num(df["natwin"].to_numpy(float), nan=0.0)[fit]
    c2, al2 = cv_corr(X, ynat, g)
    print(f"  コメント → nat_win の訓練内CV相関: {c2:+.4f} (alpha={al2})"
          f"  ← 高いほど『選手の強さを学習し直しとる』")


if __name__ == "__main__":
    main()
