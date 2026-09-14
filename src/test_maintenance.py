"""オッズ維持率 y = O_final / O(T-1) は予測できるんか。**予測が要らん手と並べる。**

=== なぜ今これか(2026-09-08) ===
R1/R3 が kill された(ROI 81% CI[76,87] / 76% CI[65,89])。**が、死に方の中身が要る。**

  T-1 → 確定 の維持率は 中央値 0.986、5%=0.232、95%=2.500。
  **|log y| の中央値 0.389 = 典型で ±48% 動く。**

R1/R3 は **T-1 のオッズで EV を計算して発火**しとった。やが払い戻されるのは**確定オッズ**や。
**決定に使った値段と精算される値段が別もん**やから、あのEVは最初から
「存在せん価格に対する期待値」やった。

→ EVで買う筋を立て直すには、**確定オッズを当てられなあかん。**その前提を先に測る。
   当てられんかったら R3'(維持率補正EV版)は構築不能=②のEV系は丸ごと閉じる。

=== 数字を見る前に決めた設計 ===

**(1) 相手は3つ。**[[principle-dumb-baseline]] + 4本目([[principle-model-residual-weak-control]])
  A **予測が要らん手**: ŷ=1.0(log y=0)。「オッズは動かん」
  B **素朴な定数**: 学習期間の平均 log y(全体のドリフト)
  C **その分野の標準的な推定量**: **オッズ十分位ごとの平均 log y**
     = 本命-大穴バイアスの既知形。人気薄ほど締切間際に動く、は誰でも知っとる
  **判定は C との差でやる。**A に勝っただけでは何も言えん

**(2) walk-forward(日で切る)。**同じレース・同じ日の艇は独立やないので、
  無作為分割はリークになる。学習=その日より前だけ

**(3) 予測に使えるんは T-1 時点で見えるもんだけ。**
  軌跡(T-15/-10/-5/-3 のオッズ)・モデル確率・レース内の総和(オーバーラウンド)・
  枠番・級別・全国勝率。**確定オッズ由来の量は1つも入れん**

**(4) 指標は RMSE と MAE を log y で。**R² も出すが、分母は C にする
  (A を分母にしたら「本命-大穴バイアスを再発見しただけ」で立派に見えてまう)

**(5) 外れ値は winsorize せん。**素で出す。裾が本体やから
  (5%が0.232、95%が2.500)。参考に 1/99% 刈った版も併記する

usage: python -u -m src.test_maintenance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import storage

RNG = np.random.default_rng(20260908)
WINDOWS = [15, 10, 5, 3, 1]
FINAL = -3          # 締切後3分 = 確定オッズ


def load():
    conn = storage.connect()
    d = pd.read_sql_query("""
        SELECT date, jcd, rno, combo, mins_to_deadline m, odds, model_p, ev,
               honmei, gyaku, racer_class, nat_win, loc_2rate
        FROM odds_timeseries WHERE bet_type='tansho'
    """, conn)
    conn.close()
    d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    d["lane"] = pd.to_numeric(d["combo"], errors="coerce")
    key = ["date", "jcd", "rno", "lane"]

    wide = d.pivot_table(index=key, columns="m", values="odds")
    wide.columns = [f"o{int(c)}" for c in wide.columns]
    meta = (d[d.m == 1].set_index(key)[["model_p", "ev", "honmei", "gyaku",
                                        "racer_class", "nat_win", "loc_2rate"]])
    X = wide.join(meta, how="inner").reset_index()
    X = X.dropna(subset=["o1", f"o{FINAL}".replace("-", "_")]
                 if False else ["o1", "o-3"])
    X["y"] = np.log(X["o-3"] / X["o1"])
    return X


def main():
    X = load()
    print(f"対 {len(X):,}艇 / {X.groupby(['date','jcd','rno']).ngroups:,}レース / "
          f"{X['date'].nunique()}日 ({X['date'].min()}-{X['date'].max()})")

    # ---- T-1 時点で見える特徴だけ ----
    X["cls"] = X["racer_class"].map({"A1": 4, "A2": 3, "B1": 2, "B2": 1}).fillna(0)
    for w in WINDOWS[:-1]:                      # 軌跡: 過去窓からT-1への変化
        c = f"o{w}"
        X[f"d{w}"] = np.log(X["o1"] / X[c]) if c in X else np.nan
    g = X.groupby(["date", "jcd", "rno"])
    X["imp"] = 1.0 / X["o1"]
    X["overround"] = g["imp"].transform("sum")   # レース内の総和(控除の見え方)
    X["imp_share"] = X["imp"] / X["overround"]
    X["n_boats"] = g["imp"].transform("size")
    X["logo1"] = np.log(X["o1"])
    X["rank"] = g["o1"].rank()

    FEA = ["logo1", "imp_share", "overround", "rank", "n_boats",
           "model_p", "ev", "honmei", "gyaku", "lane", "cls",
           "nat_win", "loc_2rate"] + [f"d{w}" for w in WINDOWS[:-1]]
    FEA = [c for c in FEA if c in X and X[c].notna().mean() > 0.5]
    print(f"特徴 {len(FEA)}本: {FEA}")

    X = X.sort_values("date").reset_index(drop=True)
    dates = sorted(X["date"].unique())
    cut = dates[len(dates) // 2]
    print(f"\nwalk-forward: 学習=その日より前 / 判定={cut} 以降")

    preds = {k: np.full(len(X), np.nan) for k in ["A", "B", "C", "M"]}
    for i, d in enumerate(dates):
        if d < cut:
            continue
        tr = (X["date"] < d).to_numpy()
        te = (X["date"] == d).to_numpy()
        if tr.sum() < 2000:
            continue
        ytr = X.loc[tr, "y"].to_numpy()
        preds["A"][te] = 0.0                                   # 動かん
        preds["B"][te] = ytr.mean()                            # 定数ドリフト
        # C: オッズ十分位ごとの平均(本命-大穴バイアスの既知形)
        q = np.quantile(X.loc[tr, "logo1"], np.linspace(0, 1, 11))
        q[0], q[-1] = -np.inf, np.inf
        btr = np.digitize(X.loc[tr, "logo1"], q[1:-1])
        bte = np.digitize(X.loc[te, "logo1"], q[1:-1])
        mu = pd.Series(ytr).groupby(btr).mean()
        preds["C"][te] = pd.Series(bte).map(mu).fillna(ytr.mean()).to_numpy()
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06,
                                          max_iter=300, l2_regularization=1.0,
                                          random_state=42)
        m.fit(X.loc[tr, FEA].to_numpy(float), ytr)
        preds["M"][te] = m.predict(X.loc[te, FEA].to_numpy(float))
        if i % 8 == 0:
            print(f"  [{d}] 学習 {tr.sum():,} / 判定 {te.sum():,}", flush=True)

    ok = ~np.isnan(preds["A"])
    y = X.loc[ok, "y"].to_numpy()
    print(f"\n判定対象 {ok.sum():,}艇")

    def sc(p, lab):
        e = y - p[ok]
        rmse, mae = np.sqrt((e ** 2).mean()), np.abs(e).mean()
        return rmse, mae

    rows = []
    for k, lab in [("A", "A 予測が要らん(y=1.0)"), ("B", "B 定数ドリフト"),
                   ("C", "C オッズ十分位平均(標準推定量)"), ("M", "M モデル")]:
        r, m_ = sc(preds[k], lab)
        rows.append((lab, r, m_))
    rC = rows[2][1]
    print(f"\n=== log y の予測誤差(小さいほど良い) ===")
    print(f"  {'相手':<30}{'RMSE':>9}{'MAE':>9}{'C比 R²':>10}")
    for lab, r, m_ in rows:
        r2 = 1 - (r ** 2) / (rC ** 2)
        print(f"  {lab:<30}{r:>9.4f}{m_:>9.4f}{r2:>10.4f}")
    print(f"\n  ※ C比R² が 0 = 標準推定量と同じ。**ここが正の値にならんかったら"
          f"「動きは読めん」**")

    # 参考: 1/99% 刈った版
    lo, hi = np.quantile(y, [0.01, 0.99])
    m2 = (y >= lo) & (y <= hi)
    print(f"\n  参考(1/99%刈り n={m2.sum():,}):")
    for k, lab in [("A", "A"), ("C", "C"), ("M", "M")]:
        e = y[m2] - preds[k][ok][m2]
        print(f"    {lab}  RMSE {np.sqrt((e ** 2).mean()):.4f}  "
              f"MAE {np.abs(e).mean():.4f}")

    # 方向だけでも当たるか(EVの符号に効くのは向き)
    print(f"\n=== 向きだけ(オッズが上がるか下がるか) ===")
    up = (y > 0).astype(int)
    for k, lab in [("C", "C 標準推定量"), ("M", "M モデル")]:
        pu = (preds[k][ok] > 0).astype(int)
        print(f"  {lab:<16}的中 {(pu == up).mean():.2%}  "
              f"(常に『上がる』と言うたら {up.mean():.2%})")


if __name__ == "__main__":
    main()
