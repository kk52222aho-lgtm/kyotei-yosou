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
# 展開(6艇の相互作用)=レース内の相対値。全て展示ST/展示タイム/地力=レース前情報でリークなし。
#  2026-07 追加: test_tenkai で 1着精度+1.2pt(単体ではベース最大の上乗せ)。エッジは市場効率で
#  存在しないと確定済(妙味ROIはsurvivorship水増しだった)ため、"1番いい予想"に振り切り採用。
#  各艇単体スペックと違い「誰が出遅れ誰が差す/まくるか」の艇間関係を唯一つかむ軸。
TENKAI = ["st_rank", "st_z", "st_gap", "inner_st_edge", "inner_pow_edge", "tt_rank", "tt_gap"]

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
] + TENKAI          # 展開(相対) ※末尾に追加=venue_codeのCAT_INDEXは不変

# HistGradientBoosting に渡すカテゴリ列のインデックス
CAT_INDEX = [FEATURES.index(c) for c in CATEGORICAL]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def _to_class_num(c) -> int:
    return CLASS_MAP.get(str(c).strip(), 0)


def _add_tenkai(df: pd.DataFrame) -> pd.DataFrame:
    """展開(レース内相対)7特徴を df に付ける。入力の行順は保持(index整合で戻す)。

    レース単位でグルーピング(date/jcd/rno が揃えばそれで、無ければ全体を1レース扱い=
    予測時の単一レース入力に対応)。展示ST→平均ST→レース平均→全体でフォールバック。
    """
    w = df.copy()
    if all(c in w.columns for c in ("date", "jcd", "rno")):
        keys = ["date", "jcd", "rno"]
    else:
        w["__rk"] = 0            # 単一レース入力(予測時) or キー無し=全体を1レース扱い
        keys = ["__rk"]

    st = pd.to_numeric(w["tenji_st"], errors="coerce") if "tenji_st" in w else pd.Series(np.nan, index=w.index)
    if "avg_st" in w:
        st = st.fillna(pd.to_numeric(w["avg_st"], errors="coerce"))
    w["_st"] = st
    w["_st"] = w["_st"].fillna(w.groupby(keys)["_st"].transform("mean")).fillna(w["_st"].mean()).fillna(0.18)
    tt = pd.to_numeric(w["tenji_time"], errors="coerce") if "tenji_time" in w else pd.Series(np.nan, index=w.index)
    w["_tt"] = tt
    w["_tt"] = w["_tt"].fillna(w.groupby(keys)["_tt"].transform("mean")).fillna(w["_tt"].mean()).fillna(6.8)
    w["_pw"] = (pd.to_numeric(w["nat_win"], errors="coerce").fillna(0.0) if "nat_win" in w
                else pd.Series(0.0, index=w.index))

    g = w.groupby(keys)
    df["st_rank"] = g["_st"].rank(method="min")                       # 速い出足の順位(1=最速)
    mu, sd = g["_st"].transform("mean"), g["_st"].transform("std").replace(0, np.nan)
    df["st_z"] = ((w["_st"] - mu) / sd).fillna(0.0)                   # レース内STの偏差
    df["st_gap"] = w["_st"] - g["_st"].transform("min")              # トップとのST差
    df["tt_rank"] = g["_tt"].rank(method="min")                       # 展示タイム順位
    df["tt_gap"] = w["_tt"] - g["_tt"].transform("min")

    # inner_*: lane昇順で"自分より内"の累積=展開の芯(差し/まくりの芽)。行順は s.index で戻す
    s = w.sort_values(keys + ["lane"])
    gs = s.groupby(keys, sort=False)
    n_inner = gs.cumcount().to_numpy()
    st_cs = gs["_st"].cumsum().to_numpy()
    inner_st_mean = np.where(n_inner > 0, (st_cs - s["_st"].to_numpy()) / np.clip(n_inner, 1, None),
                             s["_st"].to_numpy())
    pw_cummax = gs["_pw"].cummax().shift().to_numpy()                 # 自分より内の地力最大
    first = n_inner == 0
    pw_cummax = np.where(first, s["_pw"].to_numpy(), pw_cummax)
    df.loc[s.index, "inner_st_edge"] = inner_st_mean - s["_st"].to_numpy()   # 正=内より速く出る
    df.loc[s.index, "inner_pow_edge"] = s["_pw"].to_numpy() - pw_cummax      # 正=内の誰より地力上
    return df


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
    df = _add_tenkai(df)          # 展開(相対)7特徴を付与。行順は保持
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
