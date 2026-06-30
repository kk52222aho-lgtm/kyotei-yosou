"""セグメント分析：単勝・本命の回収率を条件で切り分け、利益の出るニッチを探す。

フラット張りでは控除率(~25%)に負ける。そこで「モデルのエッジが濃い条件」を探す:
  - モデル本命が1号艇か否か（インに逆らう＝穴目を見ているレース）
  - 場(venue)別
  - モデル自信度(校正済み勝率)別
  - 風の強さ別

例:
  python -m src.analyze
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .backtest import load
from .features import FEATURES, build_frame
from .train import _make_estimator
from .venues import name as venue_name


def _roi_table(g: pd.DataFrame) -> pd.Series:
    """単勝・本命でフラット100円張りした時の集計。g には hit/tansho_yen 列が必要。"""
    bets = len(g)
    hits = int(g["hit"].sum())
    ret = int(g.loc[g["hit"], "tansho_yen"].sum())
    stake = bets * 100
    return pd.Series({
        "レース": bets,
        "的中率": f"{hits / bets:.1%}" if bets else "-",
        "回収率": f"{ret / stake:.1%}" if stake else "-",
        "収支": f"{ret - stake:+,}円",
    })


def main():
    df = load()
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    tr = df["date"].to_numpy() < cut
    te = ~tr

    X = build_frame(df)[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    model = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    model.fit(X[tr], y[tr])

    test = df[te].copy()
    test["p"] = model.predict_proba(X[te])[:, 1]

    # レース単位で「モデル本命」を1行に集約
    recs = []
    for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        top = g.iloc[0]
        recs.append({
            "jcd": jcd,
            "bet_lane": int(top["lane"]),      # モデル本命の艇番
            "p0": float(top["p"]),             # 本命の校正済み勝率
            "tansho_lane": top["tansho_lane"],
            "tansho_yen": top["tansho_yen"],
            "wind": top.get("wind_speed"),
        })
    R = pd.DataFrame(recs).dropna(subset=["tansho_lane", "tansho_yen"])
    R["tansho_lane"] = R["tansho_lane"].astype(int)
    R["hit"] = R["bet_lane"] == R["tansho_lane"]

    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\nテスト期間 {cut}以降 / {len(R)}レース　＝＝ 単勝・本命フラット張りの回収率分析 ＝＝")

    print("\n【全体】")
    print(_roi_table(R).to_frame().T.to_string(index=False))

    print("\n【モデル本命の艇番別】（1号艇以外＝インに逆らう穴目）")
    t = R.groupby("bet_lane").apply(_roi_table, include_groups=False)
    print(t.to_string())

    print("\n【本命=1号艇 か否か】")
    R["contrarian"] = np.where(R["bet_lane"] == 1, "本命=1号艇", "本命≠1号艇(穴目)")
    print(R.groupby("contrarian").apply(_roi_table, include_groups=False).to_string())

    print("\n【モデル自信度(本命の校正勝率)別】")
    R["conf"] = pd.cut(R["p0"], [0, 0.3, 0.45, 0.6, 1.0],
                       labels=["~30%", "30-45%", "45-60%", "60%~"])
    print(R.groupby("conf", observed=True).apply(_roi_table, include_groups=False).to_string())

    print("\n【風速別】")
    R["wind_b"] = pd.cut(pd.to_numeric(R["wind"], errors="coerce"),
                         [-1, 2, 4, 99], labels=["~2m", "3-4m", "5m~"])
    print(R.groupby("wind_b", observed=True).apply(_roi_table, include_groups=False).to_string())

    print("\n【回収率トップ/ワースト場】(30レース以上)")
    vt = R.groupby("jcd").apply(_roi_table, include_groups=False)
    vt["_n"] = R.groupby("jcd").size()
    vt["_roi"] = R.groupby("jcd").apply(
        lambda g: g.loc[g.bet_lane == g.tansho_lane, "tansho_yen"].sum() / (len(g) * 100),
        include_groups=False)
    vt = vt[vt["_n"] >= 30].sort_values("_roi", ascending=False)
    vt.index = [f"{venue_name(j)}({j})" for j in vt.index]
    print(vt[["レース", "的中率", "回収率", "収支"]].head(5).to_string())
    print("  …")
    print(vt[["レース", "的中率", "回収率", "収支"]].tail(5).to_string())

    print("\n※回収率100%超なら理論上プラス。フラット張りの控除率は約25%。")


if __name__ == "__main__":
    main()
