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
from .features import CAT_INDEX, FEATURES, build_frame


def _make_estimator():
    """venue_code をカテゴリ特徴量として扱う HistGBM。"""
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, categorical_features=CAT_INDEX, random_state=42,
    )

MODEL_PATH = os.path.join(storage.DATA_DIR, "model.joblib")


def load_dataset() -> pd.DataFrame:
    conn = storage.connect()
    df = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    conn.close()
    return df


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
    df = load_dataset()
    if len(df) < 60:
        print(f"学習データが不足しています（{len(df)}行）。"
              "src.collect でもっと収集してください。")
        return

    feat = build_frame(df)
    X = feat[FEATURES].to_numpy(dtype=float)
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
    joblib.dump({"model": final, "features": FEATURES}, MODEL_PATH)
    print(f"\nモデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
