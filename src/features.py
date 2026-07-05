"""特徴量エンジニアリング。

出走表の生データ(dict)・DB のレコードを、モデル入力の数値ベクトルへ変換する。
学習・予測の双方で同じ関数を使い、特徴量のズレを防ぐ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 級別を数値化 (高いほど上位)
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0}

# カテゴリ特徴量（中央値補完せず、欠損は -1）。競艇は場ごとにイン勝率が大きく違う
# （実測: 大村 67.7% ↔ 戸田 41.6%）ため venue_code は最重要級の特徴量。
CATEGORICAL = ["venue_code"]

# モデルに入れる特徴量カラム
# ※2026-07 掃除: ROI-permutation(free/固定universe)＋精度permの三重確認で shuffle=完全no-op
#   (モデルが一度も分岐に使わん)だった10軸を除去 → f_count, l_count, avg_st, nat_3rate,
#   loc_3rate, motor_3rate, boat_3rate, tilt, weight_today, tenji_st。ROI無変化を検証済み。
FEATURES = [
    "venue_code",      # 場コード(jcd) ※場ごとのイン有利度を学習
    "lane",            # 艇番(コース) ※競艇では最重要。特に2連単の組順序の本体
    "class_num",
    "age", "weight",
    "nat_win", "nat_2rate",   # 全国勝率/2連率 ※単勝per-bet品質の本体
    "loc_win", "loc_2rate",   # 当地勝率/2連率
    "motor_2rate",
    "boat_2rate",
    # --- 直前情報 ---
    "tenji_time",    # 展示タイム（小さいほど速い）
    "wind_speed",    # 風速 m
    "wave_height",   # 波高 cm
]

# HistGradientBoosting に渡すカテゴリ列のインデックス
CAT_INDEX = [FEATURES.index(c) for c in CATEGORICAL]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def _to_class_num(c) -> int:
    return CLASS_MAP.get(str(c).strip(), 0)


def build_frame(rows, impute: bool = True) -> pd.DataFrame:
    """dict のリスト or DataFrame を、特徴量カラムを備えた DataFrame に整える。

    impute=True: 数値欠損を中央値補完（このフレーム内で計算）。手軽だが、
      train+test を一緒に渡すと test の中央値が混じる＝リーク。学習/検証では使うな。
    impute=False: 数値欠損は NaN のまま（HistGradientBoosting がネイティブ処理）。
      リークなし。検証(validate.py)はこちら。
    """
    df = pd.DataFrame(rows).copy()
    df["class_num"] = df["racer_class"].map(_to_class_num)
    if "jcd" in df.columns:
        df["venue_code"] = pd.to_numeric(df["jcd"], errors="coerce")
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if impute:
        df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median(numeric_only=True)).fillna(0)
    # カテゴリは欠損 -1（HistGBM が独立カテゴリとして扱う）
    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype(int)
    return df


def matrix(rows) -> np.ndarray:
    return build_frame(rows)[FEATURES].to_numpy(dtype=float)
