"""第2の検定: 文字n-gramを学習させて「語彙レベルの上限」を測る。

読み方は `docs/design_paper_comment_v2.md` で、モデルを1回も当てる前に固定した。

辞書版が holdout で落ちた後、**辞書が悪かったんか文に情報が無いんか**を分ける。
辞書は人が思いついた語しか見とらん。n-gram を学習させたら語彙レベルの上限が出る。

🚨 **holdout への2回目の照会や。**通った場合は「2回のうち1回」を割り引いて読む。

🚨 **学習を挟むと1回目に無かった漏れ方ができる。**
   せやから偽薬に「**訓練の残差を壊してから学習させる**」を足してある。
   これが通ったら配管が漏れとる。
"""
from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .scan_all import race_z
from .test_paper_comment import JCD, orthogonalized_resid

RNG = np.random.default_rng(20260923)
NBOOT = 5000
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def norm(t: str) -> str:
    return unicodedata.normalize("NFKC", t or "")


def fit_text(texts: pd.Series, y: np.ndarray, groups: np.ndarray, fit_mask: np.ndarray):
    """訓練だけで TF-IDF + Ridge。alpha は日でまとめた GroupKFold で選ぶ。"""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          min_df=20, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform(texts[fit_mask])
    ytr = y[fit_mask]
    gtr = groups[fit_mask]
    best, best_a = -np.inf, ALPHAS[0]
    gkf = GroupKFold(n_splits=4)
    for a in ALPHAS:
        sc = []
        for tri, tei in gkf.split(Xtr, ytr, gtr):
            m = Ridge(alpha=a).fit(Xtr[tri], ytr[tri])
            pred = m.predict(Xtr[tei])
            sd = pred.std()
            sc.append(0.0 if sd == 0 else float(np.corrcoef(pred, ytr[tei])[0, 1]))
        s = float(np.nanmean(sc))
        if s > best:
            best, best_a = s, a
    model = Ridge(alpha=best_a).fit(Xtr, ytr)
    Xall = vec.transform(texts)
    return model.predict(Xall), best_a, best, Xtr.shape[1]


def run(db: str, cutoff: str = "2025") -> None:
    from .scan_all import load

    df = load()
    df = df[df["jcd"] == JCD].copy()
    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    df = df.merge(oof, on=["date", "jcd", "rno", "lane"], how="left")

    # 生文も要るんで attach_comments とは別に引く
    import sqlite3
    con = sqlite3.connect(db)
    con.execute("PRAGMA busy_timeout=60000")
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
    assert len(df) == nrace * 6
    q, resid, ro, keep, okz, r2 = orthogonalized_resid(df, nrace, rid_codes)
    print(f"宮島 {nrace:,}レース / 直交化が説明した割合 {r2:.2%}")

    text = df["comment"].fillna("").map(norm)
    yr = df["date"].str[:4].to_numpy()
    days = df["dnum"].to_numpy()
    has = (text.str.len() > 0).to_numpy()

    fit_mask = (yr < cutoff) & okz & has & ~np.isnan(ro)
    print(f"学習に使う艇: {int(fit_mask.sum()):,}(訓練期間・直交化でけて・コメント在り)")
    y = np.nan_to_num(ro, nan=0.0)

    # ---- 本物 ---------------------------------------------------------------
    pred, alpha, cv, nfeat = fit_text(text, y, days, fit_mask)
    print(f"  TF-IDF 次元 {nfeat:,} / 選んだ alpha={alpha} / 訓練内CV相関 {cv:+.4f}")
    cols: dict[str, np.ndarray] = {"ngram学習スコア": race_z(np.where(has, pred, np.nan), nrace)}

    # ---- 偽薬: 訓練の残差を壊してから学習(学習を挟んだ漏れの点呼) ----------
    # 🚨 入替は**訓練のレースの中だけ**でやる。全期間で回したら holdout の
    #    残差が訓練側のラベルに混ざる = 偽薬自体が漏れる
    tr_race = (np.bincount(rid_codes, weights=(yr < cutoff).astype(float),
                           minlength=nrace) == 6)
    tr_idx = np.flatnonzero(tr_race)
    for i in range(2):
        Y = y.reshape(nrace, 6).copy()
        Y[tr_idx] = Y[RNG.permutation(tr_idx)]
        yb = Y.ravel()
        pb, ab, cb, _ = fit_text(text, yb, days, fit_mask)
        print(f"  [偽薬{i+1}] ラベル入替で学習: alpha={ab} 訓練内CV相関 {cb:+.4f}")
        cols[f"偽薬_ラベル入替で学習{i+1}"] = race_z(np.where(has, pb, np.nan), nrace)

    for i in range(2):
        cols[f"偽薬_ノイズ{i+1}"] = race_z(RNG.standard_normal(len(df)), nrace)
    cols["偽薬_natwin(カナリア)"] = race_z(df["natwin"].to_numpy(float), nrace)
    cols["点呼_着順(答え)"] = race_z(-df["fin"].to_numpy(float), nrace)

    dcodes, dun = pd.factorize(df["dnum"], sort=True)
    nd = len(dun)
    for label, mask in ((f"訓練 (< {cutoff}) ※器の点呼だけ", yr < cutoff),
                        (f"**holdout (>= {cutoff}) ← 判定はここ**", yr >= cutoff)):
        m0 = mask & okz
        m0 = (np.bincount(rid_codes, weights=m0.astype(float),
                          minlength=nrace)[rid_codes] == 6)
        names, W = [], []
        for nm, z in cols.items():
            mm = m0 & ~np.isnan(z)
            mm = (np.bincount(rid_codes, weights=mm.astype(float),
                              minlength=nrace)[rid_codes] == 6)
            if mm.sum() < 6 * 200:
                continue
            names.append(nm)
            W.append(np.bincount(dcodes, weights=np.where(mm, z, 0.0)
                                 * np.where(mm, ro, 0.0), minlength=nd))
        W = np.array(W).T
        den = np.sqrt((W ** 2).sum(0))
        dn = np.where(den == 0, np.nan, den)
        t = W.sum(0) / dn
        mx = np.empty(NBOOT)
        for a in range(0, NBOOT, 500):
            b = min(500, NBOOT - a)
            xi = RNG.standard_normal((b, W.shape[0]))
            mx[a:a + b] = np.nanmax(np.abs((xi @ W) / dn), axis=1)
        thr = float(np.percentile(mx, 95))
        print(f"\n=== {label}  {int(m0.sum() // 6):,}レース  FWER5%閾値 |t|>{thr:.2f} ===")
        for nm, tv in sorted(zip(names, t), key=lambda z: -abs(z[1])):
            print(f"  {'★通過' if abs(tv) > thr else '    '}  t={tv:+7.2f}  {nm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cutoff", default="2025")
    a = ap.parse_args()
    run(a.db, a.cutoff)


if __name__ == "__main__":
    main()
