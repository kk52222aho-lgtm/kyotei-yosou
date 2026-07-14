"""既存14に無い"新しい数字"を自作して#2の椅子を争わせる＝諦めずに第二の柱を探す。

#1=艇番(イン)は確定。他12は+0.4ptの飾り。∴今ある数字でない何かを生データから engineering:
  gap_days : この選手that前走から何日空いたか(休み明け/連闘=錆び・疲労。既存カラムに無い)
  loc_n    : この選手thaこの場で過去何走か(loc_winはあるが"本数の厚み"は別物)
  skill_rk : 今日の6人中この選手that地力何番目か(絶対勝率でなく相対=インと独立な軸)
全部リークなし(過去のみ)。14ベースに1個ずつ足し1着精度の上乗せを測る=#2の理由になるやつthaあるか。
確定払戻・walk-forward。

例: python -u -m src.test_new2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator
from .validate import SELECTION_YEARS

NEW = ["gap_days", "loc_n", "skill_rk"]


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
    dt = pd.to_datetime(df["date"], format="%Y%m%d")
    df["_dt"] = dt
    # 出走間隔: reg毎に前回出走日との差(日)。初出はNaN→30で埋め、0-60にclip
    df = df.sort_values(["reg", "_dt", "rno"]).reset_index(drop=True)
    prev = df.groupby("reg")["_dt"].shift()
    df["gap_days"] = (df["_dt"] - prev).dt.days.clip(0, 60).fillna(30).astype(float)
    # 当地経験本数: (reg,jcd)毎の過去出走数(自分を除く=leak-free)
    df = df.sort_values(["reg", "jcd", "_dt", "rno"]).reset_index(drop=True)
    df["loc_n"] = df.groupby(["reg", "jcd"]).cumcount().astype(float)
    # レース内地力ランク: nat_win のレース内降順順位(1=最強)。相対=インと独立
    df["skill_rk"] = df.groupby(["date", "jcd", "rno"])["nat_win"].rank(ascending=False, method="min").astype(float)
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
    print("新自作特徴の充足/分布:")
    print(f"  gap_days 中央{df['gap_days'].median():.0f}日  loc_n 中央{df['loc_n'].median():.0f}本  skill_rk 平均{df['skill_rk'].mean():.1f}\n")
    base = acc(df, [])
    print(f"  ベース14特徴            : {base:.2f}%   (常にイン買い≈55.1%)")
    for f in NEW:
        a = acc(df, [f])
        label = {"gap_days": "＋出走間隔", "loc_n": "＋当地経験本数", "skill_rk": "＋レース内地力ランク"}[f]
        print(f"  {label:20s}: {a:.2f}%   ({a-base:+.2f}pt)")
    allf = acc(df, NEW)
    print(f"  ＋3つ全部             : {allf:.2f}%   ({allf-base:+.2f}pt)")
    print("\n読み: どれかthat+0.4pt(既存12の合計)を明確に超えれば、それthaが本物の第二の柱=#2の理由。")
    print("      全部+0.1pt未満なら第二の柱も無し=イン一本で確定。確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
