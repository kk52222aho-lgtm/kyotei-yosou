"""オリジナル展示(一周タイム)を特徴量に足すと売り物のpick精度が上がるか。

既存モデル(features.FEATURES, HistGBM)に対し、レース内lap順位(と まわり足順位)を
追加した時の OOS 本命的中率(race top-1)/AUC の増分を、レース単位GroupKFoldで測る。
的中率の差はレース単位ペアブートストラップで有意性を判定。

生値は場横断非比較なので特徴量はレース内順位で入れる。oriten を収集済みの日のみ対象。

使い方: python test_oriten_feature.py
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import storage
from features import FEATURES, CAT_INDEX, build_frame

EXTRA = ["lap_rank", "mawariashi_rank"]


def race_topk_accuracy(df, proba):
    tmp = df.copy(); tmp["p"] = proba
    hit = total = 0
    for _, g in tmp.groupby(["date", "jcd", "rno"]):
        pred = g.loc[g["p"].idxmax(), "lane"]
        act = g.loc[g["win"] == 1, "lane"]
        if len(act) == 0:
            continue
        total += 1
        hit += int(pred == act.iloc[0])
    return hit / total if total else 0.0


def lane1_baseline(df):
    hit = total = 0
    for _, g in df.groupby(["date", "jcd", "rno"]):
        act = g.loc[g["win"] == 1, "lane"]
        if len(act) == 0:
            continue
        total += 1
        hit += int(act.iloc[0] == 1)
    return hit / total if total else 0.0


def make_est():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, categorical_features=CAT_INDEX, random_state=42,
    )


def load():
    conn = storage.connect()
    o = pd.read_sql(
        "SELECT date,jcd,rno,lane,lap,halflap,mawariashi FROM oriten", conn)
    dates = tuple(sorted(o["date"].unique()))
    q = ("SELECT * FROM entries WHERE win IS NOT NULL AND date IN "
         f"({','.join('?'*len(dates))})")
    e = pd.read_sql(q, conn, params=dates)
    conn.close()
    df = e.merge(o, on=["date", "jcd", "rno", "lane"], how="inner")
    df["lap_metric"] = df["lap"].fillna(df["halflap"])
    key = ["date", "jcd", "rno"]
    df["lap_rank"] = df.groupby(key)["lap_metric"].rank(method="first", ascending=True)
    df["mawariashi_rank"] = df.groupby(key)["mawariashi"].rank(method="first", ascending=True)
    # 連続エンコード: レース内z化 と 最速艇との差(gap)
    g = df.groupby(key)["lap_metric"]
    df["lap_z"] = (df["lap_metric"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df["lap_gap"] = df["lap_metric"] - g.transform("min")
    return df


ORITEN_COLS = ["lap_rank", "mawariashi_rank", "lap_z", "lap_gap"]


def oof_proba(df, cols):
    frame = build_frame(df, impute=False)
    for c in ORITEN_COLS:
        frame[c] = pd.to_numeric(df[c].values, errors="coerce")
    X = frame[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    groups = (df["date"] + df["jcd"] + df["rno"].astype(str)).values
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = make_est()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def race_hits(df, proba):
    """各レース: 予測1位が実際1着か (1/0) を dict{key:hit} で返す。"""
    tmp = df.copy(); tmp["p"] = proba
    out = {}
    for k, g in tmp.groupby(["date", "jcd", "rno"]):
        pred = g.loc[g["p"].idxmax(), "lane"]
        act = g.loc[g["win"] == 1, "lane"]
        if len(act):
            out[k] = int(pred == act.iloc[0])
    return out


def main():
    df = load()
    nrace = df.groupby(["date", "jcd", "rno"]).ngroups
    print(f"対象: {len(df)}艇 / {nrace}レース / {df['date'].nunique()}日 "
          f"(oriten収集済み日)")

    base = oof_proba(df, FEATURES)
    base1 = lane1_baseline(df)
    acc_b = race_topk_accuracy(df, base)
    auc_b = roc_auc_score(df["win"], base)
    print(f"\n1号艇ベタ本命的中率(ベンチ) = {base1:.4f}")
    print(f"base(既存FEATURESのみ): 本命的中={acc_b:.4f} AUC={auc_b:.4f}")

    variants = {
        "+lap_rank,mawari_rank": FEATURES + ["lap_rank", "mawariashi_rank"],
        "+lap_z,lap_gap":        FEATURES + ["lap_z", "lap_gap"],
        "+全部(rank+z+gap)":     FEATURES + ["lap_rank", "mawariashi_rank", "lap_z", "lap_gap"],
    }
    hb = race_hits(df, base)
    rng = np.random.default_rng(0)
    for name, cols in variants.items():
        aug = oof_proba(df, cols)
        acc_a = race_topk_accuracy(df, aug)
        auc_a = roc_auc_score(df["win"], aug)
        ha = race_hits(df, aug)
        keys = [k for k in hb if k in ha]
        diff = np.array([ha[k] - hb[k] for k in keys], dtype=float)
        boot = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        flag = "＋改善" if lo > 0 else ("×悪化" if hi < 0 else "±差なし")
        print(f"  [{name:22s}] 的中Δ={acc_a-acc_b:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"AUCΔ={auc_a-auc_b:+.4f} {flag}")


if __name__ == "__main__":
    main()
