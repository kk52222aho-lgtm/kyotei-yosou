"""モデル学習。

各艇が「1着になる確率」を予測する二値分類器を学習する（まずシンプルに）。
予測時はレース内の6艇の確率を相対比較してランク付けする。

例:
  python -m src.train
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from . import storage
from .features import (CAT_INDEX, FEATURES, FEATURES_SCAN, SCAN_COLS,
                       build_frame)


def _make_estimator():
    """venue_code をカテゴリ特徴量として扱う HistGBM。"""
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, categorical_features=CAT_INDEX, random_state=42,
    )

MODEL_PATH = os.path.join(storage.DATA_DIR, "model.joblib")


def load_dataset() -> tuple[pd.DataFrame, bool]:
    """(学習用の行, 走査24本が使えるか) を返す。

    走査列は表 `scan_features` から **LEFT JOIN で借りる**。計算はここに置かん——
    置いたら本番と2か所になってズレる。出どころを1つにしとくんが skew を
    構造的に防ぐ唯一の形や([[insight_empty_column_blames_the_fetcher]])。
    """
    conn = storage.connect()
    df = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    ok = False
    try:
        cols = ", ".join(f"s.{c}" for c in SCAN_COLS)
        sf = pd.read_sql_query(
            f"SELECT s.date, s.jcd, s.rno, s.lane, {cols}, s.spec_sha"
            " FROM scan_features s", conn)
    except Exception as ex:                                  # noqa: BLE001
        print(f"  走査表が読めん({type(ex).__name__})。既存21列で学習する")
        sf = None
    conn.close()
    if sf is not None and len(sf):
        nsha = sf["spec_sha"].nunique()
        df["jcd"] = df["jcd"].astype(str).str.zfill(2)
        sf["jcd"] = sf["jcd"].astype(str).str.zfill(2)
        k = ["date", "jcd", "rno", "lane"]
        n0 = len(df)
        df = df.merge(sf.drop(columns=["spec_sha"]), on=k, how="left")
        cov = df[SCAN_COLS].notna().any(axis=1).mean()
        rowcov = df["f00"].notna().mean() if "f00" in df else 0.0
        assert len(df) == n0, "merge で行が増えた"
        print(f"  走査表: {len(sf):,}艇 / spec_sha {nsha}種 / "
              f"学習行に貼れた割合 {cov:.1%}")
        # 🚨 門: 版が複数混じっとる / 覆いが薄い なら走査は使わん。
        #    黙って NaN を食わせたら「学習は欠損で育ったのに本番は値が来る」形になる
        if nsha != 1:
            print("  🚨 spec_sha が複数。`build_scan_features --all` で揃えるまで使わん")
        elif cov < 0.97:
            print(f"  🚨 覆いが {cov:.1%}(門は97%)。走査は使わん")
        else:
            ok = True
    return df, ok


def race_topk_accuracy(df: pd.DataFrame, proba: np.ndarray) -> float:
    """各レースで予測1位の艇が実際に1着だった割合（本命的中率）。"""
    tmp = df.copy()
    tmp["p"] = proba
    hit = total = 0
    for _, g in tmp.groupby(["date", "jcd", "rno"]):
        pred_lane = g.loc[g["p"].idxmax(), "lane"]
        actual = g.loc[g["win"] == 1, "lane"]
        if len(actual) == 0:
            continue
        total += 1
        if pred_lane == actual.iloc[0]:
            hit += 1
    return hit / total if total else 0.0


def lane1_baseline(df: pd.DataFrame) -> float:
    """常に1号艇を本命にしたときの的中率（競艇の強力なベースライン）。"""
    hit = total = 0
    for _, g in df.groupby(["date", "jcd", "rno"]):
        actual = g.loc[g["win"] == 1, "lane"]
        if len(actual) == 0:
            continue
        total += 1
        if actual.iloc[0] == 1:
            hit += 1
    return hit / total if total else 0.0


def main():
    df, use_scan = load_dataset()
    if len(df) < 60:
        print(f"学習データが不足しています（{len(df)}行）。"
              "src.collect でもっと収集してください。")
        return

    cols = FEATURES_SCAN if use_scan else FEATURES
    print(f"  特徴量 {len(cols)}列" + ("(走査24本入り)" if use_scan else "(既存のみ)"))
    feat = build_frame(df)
    X = feat[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)

    # 時系列を尊重して日付で train/test 分割
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.8)] if len(dates) > 4 else dates[-1]
    tr = df["date"].to_numpy() < cut
    te = ~tr
    if te.sum() == 0:  # 全期間が同日などのフォールバック
        tr = np.ones(len(df), bool); tr[::5] = False; te = ~tr

    model = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    model.fit(X[tr], y[tr])

    p_te = model.predict_proba(X[te])[:, 1]
    try:
        auc = roc_auc_score(y[te], p_te)
    except ValueError:
        auc = float("nan")
    ll = log_loss(y[te], p_te, labels=[0, 1])
    acc = race_topk_accuracy(df[te], p_te)
    base = lane1_baseline(df[te])
    verdict = "✓ ベース超え" if acc > base else "× ベース未達（要改善）"

    print("=== 評価 (test) ===")
    print(f"  レース数(test): {df[te].groupby(['date','jcd','rno']).ngroups}")
    print(f"  AUC            : {auc:.3f}")
    print(f"  LogLoss        : {ll:.3f}")
    print(f"  本命的中率     : {acc:.1%}")
    print(f"  1号艇ベタ買い  : {base:.1%}   → {verdict}")

    # 本番モデルは全データで再学習
    final = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    final.fit(X, y)
    # 🚨 2026-09-27: ここで**壊れた束をディスクに残した**。X は45列で当てはめたのに
    #    束には `features: FEATURES`(21列)を書いとって、予測側が45列モデルに
    #    21列を食わせる形になっとった。門が無かったから気付いたんは後や。
    #    保存する前に**宣言した列数とモデルが食う列数を突き合わせる**。
    nin = getattr(final, "n_features_in_", None)
    if nin is not None and nin != len(cols):
        raise RuntimeError(
            f"束が壊れとる: 宣言 {len(cols)}列 / モデルが食う {nin}列。保存せん")
    # 上書きする前に控えを取る。戻せんようにせん
    if os.path.exists(MODEL_PATH):
        import shutil
        import time as _t
        bak = f"{MODEL_PATH}.bak_{_t.strftime('%Y%m%d_%H%M')}"
        shutil.copy2(MODEL_PATH, bak)
        print(f"  控え: {bak}")
    joblib.dump({"model": final, "features": cols}, MODEL_PATH)
    print(f"\nモデルを保存しました: {MODEL_PATH}  ({len(cols)}列)")


if __name__ == "__main__":
    main()
