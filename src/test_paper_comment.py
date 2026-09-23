"""宮島の選手コメントが、既存の盤の外に情報を持っとるかを測る。

**読み方は `docs/design_paper_comment.md` で、スコアを1回も出す前に固定した。**
ここはそれを実装しただけ。判定の規則をここで足したら事後選択になる。

相手・統計は `scan_all.py` と**同じもん**にしてある(道具を揃えな比較でけへん):

- 相手 = **既存14特徴のレース内順位 + モデル確率 へ直交化した残差**。
  モデルの生残差やと「モデルが手持ちを使い切っとらん」分を拾う
  → [[insight_model_residual_is_a_weak_control]]
- 統計 = 日クラスタの multiplier bootstrap で max|t| の95%点(FWER)

偽薬を**同じ配管に**流す。1本でも通ったら結果は読まん。
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

from .paper_features import score as cmt_score_of
from .scan_all import load, race_z

RNG = np.random.default_rng(20260923)
NBOOT = 5000
JCD = "17"

BASE_RANK = ["lane", "class_num", "age", "weight",
             "nat_win", "nat_2rate", "loc_win", "loc_2rate",
             "motor_2rate", "boat_2rate", "tenji_time",
             "tt_rank", "tt_gap", "inner_pow_edge"]


def attach_comments(df: pd.DataFrame, db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    con.execute("PRAGMA busy_timeout=60000")
    pc = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, comment FROM paper_comment WHERE jcd=?",
        con, params=(JCD,))
    con.close()
    pc["jcd"] = pc["jcd"].astype(str).str.zfill(2)
    pc["rno"] = pc["rno"].astype(int)
    pc["lane"] = pc["lane"].astype(int)
    sc = pc["comment"].map(cmt_score_of).apply(pd.Series)
    pc = pd.concat([pc.drop(columns=["comment"]), sc], axis=1)
    return df.merge(pc, on=["date", "jcd", "rno", "lane"], how="left")


def orthogonalized_resid(df: pd.DataFrame, nrace: int, rid_codes: np.ndarray):
    """scan_all.scan と同じ手順で、直交化した残差を返す。"""
    from .features import build_frame

    p = df["p"].to_numpy(dtype=float)
    win = df["win"].to_numpy(dtype=float)
    have = ~np.isnan(p) & ~np.isnan(win)
    cnt = np.bincount(rid_codes, weights=have.astype(float), minlength=nrace)
    keep = cnt[rid_codes] == 6
    psum = np.bincount(rid_codes, weights=np.where(keep, p, 0.0), minlength=nrace)
    q = np.where(keep, p / np.maximum(psum[rid_codes], 1e-12), np.nan)
    resid = np.where(keep, win - q, np.nan)

    bf = build_frame(df, impute=False)
    Z = [race_z(pd.to_numeric(bf[c], errors="coerce").to_numpy(float), nrace)
         for c in BASE_RANK]
    Z.append(np.where(keep, q, np.nan))
    Z = np.column_stack(Z)
    okz = keep & ~np.isnan(Z).any(axis=1)
    okz = (np.bincount(rid_codes, weights=okz.astype(float),
                       minlength=nrace)[rid_codes] == 6)
    Zk = Z[okz]
    beta, *_ = np.linalg.lstsq(Zk, resid[okz], rcond=None)
    ro = np.full(len(df), np.nan)
    ro[okz] = resid[okz] - Zk @ beta
    r2 = 1 - np.var(ro[okz]) / np.var(resid[okz])
    return q, resid, ro, keep, okz, r2


def shuffle_races(vals: np.ndarray, nrace: int, rng) -> np.ndarray:
    """レース塊ごと丸ごと入替。**レース内の並びは壊さん**ので、
    『レース内順位の形』は同じまま中身だけ別レースのもんになる。"""
    V = vals.reshape(nrace, 6).copy()
    return V[rng.permutation(nrace)].ravel()


def run(db: str, cutoff: str = "2025") -> None:
    df = load()
    df = df[df["jcd"] == JCD].copy()
    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    df = df.merge(oof, on=["date", "jcd", "rno", "lane"], how="left")
    df = attach_comments(df, db)
    df = df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)

    rid_codes, uniq = pd.factorize(df["rid"], sort=False)
    nrace = len(uniq)
    assert len(df) == nrace * 6, f"行数が6の倍数やない: {len(df)}"
    print(f"宮島 {nrace:,} レース / {len(df):,} 艇")
    have_c = df["cmt_score"].notna()
    print(f"コメントが付いた艇: {have_c.sum():,} ({have_c.mean():.1%})")
    full6 = np.bincount(rid_codes, weights=have_c.to_numpy(float),
                        minlength=nrace)[rid_codes] == 6
    print(f"6艇そろってコメントが在るレース: {int(full6.sum() // 6):,}")

    q, resid, ro, keep, okz, r2 = orthogonalized_resid(df, nrace, rid_codes)
    print(f"既存特徴の順位が残差から説明した割合: {r2:.2%}")

    # ---- 測る列(本物2本 + 偽薬6本)。同じ配管に流す -----------------------
    cols: dict[str, np.ndarray] = {}
    cols["cmt_score"] = race_z(df["cmt_score"].to_numpy(float), nrace)
    cols["cmt_len"] = race_z(df["cmt_len"].to_numpy(float), nrace)
    for i in range(3):
        cols[f"偽薬_ノイズ{i+1}"] = race_z(RNG.standard_normal(len(df)), nrace)
    for i in range(2):
        cols[f"偽薬_塊入替{i+1}"] = shuffle_races(cols["cmt_score"], nrace, RNG)
    cols["偽薬_natwin(カナリア)"] = race_z(df["natwin"].to_numpy(float), nrace)
    cols["点呼_着順(答え)"] = race_z(-df["fin"].to_numpy(float), nrace)

    dcodes, dun = pd.factorize(df["dnum"], sort=True)
    nd = len(dun)
    yr = df["date"].str[:4].to_numpy()

    for label, mask in (("訓練 (< %s)" % cutoff, yr < cutoff),
                        ("holdout (>= %s)" % cutoff, yr >= cutoff)):
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
        t = W.sum(0) / np.where(den == 0, np.nan, den)
        dn = np.where(den == 0, np.nan, den)
        mx = np.empty(NBOOT)
        for a in range(0, NBOOT, 500):
            b = min(500, NBOOT - a)
            xi = RNG.standard_normal((b, W.shape[0]))
            mx[a:a + b] = np.nanmax(np.abs((xi @ W) / dn), axis=1)
        thr = float(np.percentile(mx, 95))
        nr = int(m0.sum() // 6)
        print(f"\n=== {label}  {nr:,}レース  FWER5%閾値 |t|>{thr:.2f} ===")
        for nm, tv in sorted(zip(names, t), key=lambda z: -abs(z[1])):
            mark = "★通過" if abs(tv) > thr else "    "
            print(f"  {mark}  t={tv:+7.2f}  {nm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cutoff", default="2025")
    a = ap.parse_args()
    run(a.db, a.cutoff)


if __name__ == "__main__":
    main()
