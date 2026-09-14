"""R3' を後ろ向きに測る。**事前登録やない。書く価値があるかを決めるための探索。**

=== 経緯 ===
- R3(T-1のオッズでEV)は kill。ROI 76%。死因は発火艇の維持率 0.532
- 天井(**確定オッズを完璧に知っとる**場合)は閾値を上げると 102.7%〜150.7% まで伸びた。
  ただし **CI下限は一本も100%を超えん**(最良で91.6%)
- 維持率は読める(C比R² +0.40、向き 75.5%)。**やが RMSE は log で 0.485
  = 典型の誤差が ×1.62 倍**や。EV閾値2.50に対してこの誤差はでかい

→ **天井のどれだけが、実際の予測誤差を入れて残るか。**これが R3' を書く価値の全てや。

  ほとんど残らん → EV系の枝を閉じる。事前登録を書かん
  残る          → 新規事前登録を書いて N を 0 から積む

=== 3つ並べる(全部同じレース・同じ買い方) ===
  実測  EV = p × O(T-1)          ← R3。死んだ側
  R3'   EV = p × O(T-1) × exp(ŷ) ← 維持率モデルで確定オッズを予測
  天井  EV = p × O(確定)          ← 実戦では不可能

維持率モデルは walk-forward(その日より前だけで学習)。相手に **1号艇ベタ買い**を必ず置く。

usage: python -u -m src.test_r3prime
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import storage
from .test_ev_ceiling import load, ci

WINDOWS = [15, 10, 5, 3]
THRS = [1.15, 1.30, 1.50, 1.75, 2.00, 2.50]


def main():
    X = load().dropna(subset=["o1", "model_p", "o-3"]).copy()
    g = X.groupby(["date", "jcd", "rno"])
    X["p"] = X["model_p"] / g["model_p"].transform("sum")
    X["y"] = np.log(X["o-3"] / X["o1"])
    X["logo1"] = np.log(X["o1"])
    X["imp"] = 1 / X["o1"]
    X["overround"] = g["imp"].transform("sum")
    X["imp_share"] = X["imp"] / X["overround"]
    X["rank"] = g["o1"].rank()
    for w in WINDOWS:
        c = f"o{w}"
        X[f"d{w}"] = np.log(X["o1"] / X[c]) if c in X else np.nan
    FEA = [c for c in ["logo1", "imp_share", "overround", "rank", "p",
                       "honmei", "gyaku", "lane"] + [f"d{w}" for w in WINDOWS]
           if c in X and X[c].notna().mean() > 0.5]
    X = X.sort_values("date").reset_index(drop=True)
    dates = sorted(X["date"].unique())
    cut = dates[len(dates) // 2]
    print(f"盤 {len(X):,}艇 / {X.groupby(['date','jcd','rno']).ngroups:,}レース")
    print(f"特徴 {len(FEA)}本。walk-forward: 判定={cut} 以降\n")

    X["yhat"] = np.nan
    for d in dates:
        if d < cut:
            continue
        tr = (X["date"] < d).to_numpy()
        te = (X["date"] == d).to_numpy()
        if tr.sum() < 2000:
            continue
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06,
                                          max_iter=300, l2_regularization=1.0,
                                          random_state=42)
        m.fit(X.loc[tr, FEA].to_numpy(float), X.loc[tr, "y"].to_numpy())
        X.loc[te, "yhat"] = m.predict(X.loc[te, FEA].to_numpy(float))

    S = X.dropna(subset=["yhat"]).copy()
    S["ev_t1"] = S["p"] * S["o1"]
    S["ev_r3p"] = S["p"] * S["o1"] * np.exp(S["yhat"])
    S["ev_fin"] = S["p"] * S["o-3"]
    nr = S.groupby(["date", "jcd", "rno"]).ngroups
    print(f"判定対象 {len(S):,}艇 / {nr:,}レース "
          f"({S['date'].min()}-{S['date'].max()})")
    print(f"維持率予測の out-of-sample RMSE(log) = "
          f"{np.sqrt(((S['y'] - S['yhat']) ** 2).mean()):.4f}\n")

    b1 = S[S["lane"] == 1]
    ret1 = np.where(b1["won"] == 1, b1["tansho_yen"], 0.0)
    print(f"相手(予測が要らん手): 1号艇ベタ買い {len(b1):,}点  "
          f"回収 {ret1.sum() / (len(b1) * 100):.1%}\n")

    print("  " + "閾値".rjust(6) + "  "
          + "実測 EV=p*O(T-1)".ljust(24)
          + "R3prime EV=p*Ohat".ljust(24)
          + "天井 EV=p*O(確定)".ljust(24))
    for t in THRS:
        line = f"  {t:>6.2f}  "
        for col in ["ev_t1", "ev_r3p", "ev_fin"]:
            gg = S.groupby(["date", "jcd", "rno"])
            top = S.loc[gg[col].idxmax()]
            top = top[top[col] >= t]
            if len(top) < 40:
                line += f"{'(発火' + str(len(top)) + ')':<26}"
                continue
            ret = np.where(top["won"] == 1, top["tansho_yen"], 0.0)
            roi = ret.sum() / (len(top) * 100)
            lo, hi = ci(ret, np.ones(len(top)), top["date"].to_numpy())
            line += f"{len(top):>5,}件 {roi:>6.1%}[{lo:>5.0%},{hi:>4.0%}]  "
        print(line)
    print("  ※ 単勝の払戻率は75%(2026-09-08訂正。旧記載85%は誤り)。金になるのはCI下限が100%を超えた時だけ")
    print("  ※ これは後ろ向き。事前登録やない。「合格」は名乗れん")


if __name__ == "__main__":
    main()
