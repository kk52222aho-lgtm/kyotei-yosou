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
FEATURES = [
    "venue_code",      # 場コード(jcd) ※場ごとのイン有利度を学習
    "lane",            # 艇番(コース) ※競艇では最重要
    "class_num",
    "age", "weight",
    "f_count", "l_count", "avg_st",
    "nat_win", "nat_2rate", "nat_3rate",
    "loc_win", "loc_2rate", "loc_3rate",
    "motor_2rate", "motor_3rate",
    "boat_2rate", "boat_3rate",
    # --- 直前情報 ---
    "tenji_time",    # 展示タイム（小さいほど速い）
    "tilt",          # チルト角
    "weight_today",  # 当日体重
    "tenji_st",      # ST展示（フライングは負値）
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
