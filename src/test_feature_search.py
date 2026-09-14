"""新特徴の総当たり探索: 既存14特徴に"今節/近走の調子"を足して上がるやつを探す。

既存FEATURESは飽和(死に軸掃除/生ID掘りで追加ゼロ実証済)。唯一未テスト＝累積率が隠す
"今節あってるか/近走の勢い"。候補をleak-freeで作り、ベース vs +候補で
1着精度＋妙味(本命≠1号)単勝ROIの上乗せを測る。上がれば新エッジ、ゼロなら飽和に加え正式に閉じる。

候補(全てそのレース前のみ＝リークなし):
  mm_win/mm_2/mm_3 : 今節モーター 勝率/2連率/3連率(同jcd+motor+節)
  rm_win/rm_2      : 今節選手   勝率/2連率(同jcd+reg+節)
  rr_win/rr_2      : 近走選手   直近10走 勝率/2連率(reg横断)

例: python -u -m src.test_feature_search
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator
from .validate import SELECTION_YEARS

CANDS = ["mm_win", "mm_2", "mm_3", "rm_win", "rm_2", "rr_win", "rr_2"]


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    df["yr"] = df["date"].str[:4]
    return df


def _prior_rate(df, keys, col):
    g = df.groupby(keys, sort=False)
    before_sum = g[col].cumsum().values - df[col].values
    before_n = g.cumcount().values
    return np.where(before_n >= 2, before_sum / np.clip(before_n, 1, None), np.nan)


def add_features(df):
    mb = pd.read_csv("data/motor_boat.csv", dtype={"date": str, "jcd": str})
    mb["jcd"] = mb["jcd"].str.zfill(2)
    df = df.merge(mb[["date", "jcd", "rno", "lane", "motor_no"]],
                  on=["date", "jcd", "rno", "lane"], how="left")
    dts = df[["jcd", "date"]].drop_duplicates().copy()
    dts["d"] = pd.to_datetime(dts["date"], format="%Y%m%d")
    dts = dts.sort_values(["jcd", "d"])
    dts["gap"] = dts.groupby("jcd")["d"].diff().dt.days.fillna(99)
    dts["meet_id"] = dts.groupby("jcd")["gap"].transform(lambda s: (s > 1).cumsum())
    df = df.merge(dts[["jcd", "date", "meet_id"]], on=["jcd", "date"], how="left")
    f = df["finish"].fillna(99)
    df["_w"] = (f == 1).astype(float)
    df["_2"] = ((f >= 1) & (f <= 2)).astype(float)
    df["_3"] = ((f >= 1) & (f <= 3)).astype(float)
    df = df.sort_values(["jcd", "motor_no", "meet_id", "date", "rno"]).reset_index(drop=True)
    df["mm_win"] = _prior_rate(df, ["jcd", "motor_no", "meet_id"], "_w")
    df["mm_2"] = _prior_rate(df, ["jcd", "motor_no", "meet_id"], "_2")
    df["mm_3"] = _prior_rate(df, ["jcd", "motor_no", "meet_id"], "_3")
    df = df.sort_values(["jcd", "reg", "meet_id", "date", "rno"]).reset_index(drop=True)
    df["rm_win"] = _prior_rate(df, ["jcd", "reg", "meet_id"], "_w")
    df["rm_2"] = _prior_rate(df, ["jcd", "reg", "meet_id"], "_2")
    df = df.sort_values(["reg", "date", "jcd", "rno"]).reset_index(drop=True)
    g = df.groupby("reg", sort=False)
    df["rr_win"] = g["_w"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()).values
    df["rr_2"] = g["_2"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()).values
    return df


def fit_eval(df, extra):
    cols = FEATURES + extra
    X = pd.concat([build_frame(df, impute=False)[FEATURES].reset_index(drop=True),
                   df[CANDS].reset_index(drop=True)], axis=1)[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    acc, roi_num, roi_den = [], 0.0, 0
    for yv in [v for v in sorted(df["yr"].unique()) if v not in SELECTION_YEARS]:
        te = (df["yr"] == yv).to_numpy()
        tr = (df["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        out = df[te].copy()
        out["p"] = m.predict_proba(X[te])[:, 1]
        for _, g in out.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if pd.isna(top["tansho_lane"]):
                continue
            acc.append(int(top["tansho_lane"]) == int(top["lane"]))
            if int(top["lane"]) != 1:
                roi_den += 1
                if int(top["tansho_lane"]) == int(top["lane"]):
                    roi_num += float(top["tansho_yen"])
    return np.mean(acc) * 100, (roi_num / (roi_den * 100) * 100 if roi_den else 0)


def main():
    df = load()
    df = add_features(df)
    print("候補特徴の充足率:")
    for c in CANDS:
        print(f"  {c}: {df[c].notna().mean()*100:.0f}%")
    print("\n=== 上乗せ探索（1着精度 / 妙味単勝ROI）===")
    base = fit_eval(df, [])
    print(f"  ベース(14特徴)      : 1着{base[0]:.1f}% / 妙味単勝{base[1]:.0f}%")
    allc = fit_eval(df, CANDS)
    print(f"  +今節/近走 全候補7個 : 1着{allc[0]:.1f}% / 妙味単勝{allc[1]:.0f}%")
    print(f"\n  -> 上乗せ: 1着 {allc[0]-base[0]:+.1f}pt / 妙味単勝 {allc[1]-base[1]:+.0f}pt")
    if allc[0] - base[0] >= 0.5 or allc[1] - base[1] >= 5:
        print("  ** 上がった！ 個別に絞って本物か検証する価値あり。")
    else:
        print("  今節/近走も上乗せ無し＝飽和に加えて正式に閉じる（既存14特徴が既に天井）。")
    print("\n※確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
