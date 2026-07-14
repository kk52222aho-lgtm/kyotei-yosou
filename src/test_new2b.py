"""#2の椅子・第2弾: まだ試してない相対軸をengineeringして争わせる。

第1弾(出走間隔/当地本数/地力ランク)は全部~0。残る新軸:
  motor_rk : レース内のモーター2連率順位(弱いモーター場での最強機=相対機力)
  weight_rk: レース内の体重順位(軽い=出足・相対)
  age_in   : 年齢×イン(1号) 相互作用(ベテランの逃げ切り力)
全部リークなし。14ベースに1個ずつ+全部で1着精度上乗せ。+0.4pt超えれば#2の理由。
確定払戻・walk-forward。

例: python -u -m src.test_new2b
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator
from .validate import SELECTION_YEARS

NEW = ["motor_rk", "weight_rk", "age_in"]


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def add_new(df):
    key = ["date", "jcd", "rno"]
    df["motor_rk"] = df.groupby(key)["motor_2rate"].rank(ascending=False, method="min").astype(float)
    df["weight_rk"] = df.groupby(key)["weight"].rank(ascending=True, method="min").astype(float)  # 軽い=1位
    df["age_in"] = df["age"].fillna(df["age"].median()) * (df["lane"] == 1).astype(float)
    return df


def acc(df, extra):
    cols = FEATURES + extra
    base = build_frame(df, impute=False)[FEATURES].reset_index(drop=True)
    X = pd.concat([base, df[NEW].reset_index(drop=True)], axis=1)[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    a = []
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
            a.append(int(top["tansho_lane"]) == int(top["lane"]))
    return np.mean(a) * 100


def main():
    df = load()
    df = add_new(df)
    base = acc(df, [])
    print(f"  ベース14特徴          : {base:.2f}%   (常にイン買い≈55.1%)")
    for f in NEW:
        a = acc(df, [f])
        label = {"motor_rk": "＋相対モーター順位", "weight_rk": "＋相対体重順位", "age_in": "＋年齢×イン"}[f]
        print(f"  {label:18s}: {a:.2f}%   ({a-base:+.2f}pt)")
    allf = acc(df, NEW)
    print(f"  ＋3つ全部           : {allf:.2f}%   ({allf-base:+.2f}pt)")
    print("\n読み: どれかthat+0.4ptを明確に超えれば#2の理由。全部~0なら第二の柱は存在せず=イン一本で最終確定。")
    print("※確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
