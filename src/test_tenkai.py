"""展開(6艇の相互作用)をモデル特徴に入れたら壁が動くか＝走る姿を描いて探す。

今までの特徴は全部"各艇単体の絶対スペック"。だがモデルは1レースを6行バラバラに予測して
正規化するだけ＝艇同士の関係(展開)を構造的に見てない。競艇の着順は個の合計でなく1マールの
相互作用: 1号が出遅れれば2号が差す、内が全部遅けりゃ4号がまくる。
∴"レース内の相対値"を作って足す(全部展示ST/展示タイム=レース前情報でリークなし):
  st_rank/st_z/st_gap : このレースで何番目に速く出るか(主導権)
  inner_st_edge       : 内の艇より速く出るか=差し/まくりの芽(正=内を抜く展開)
  inner_pow_edge      : 内の艇より地力が上か
  tt_rank/tt_gap      : 展示タイム(一周の速さ)のレース内順位
ベース14 vs +展開 で 1着精度 / 妙味(本命≠1号)単勝ROI の上乗せを測る。確定払戻・walk-forward。

例: python -u -m src.test_tenkai
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator
from .validate import SELECTION_YEARS

TENKAI = ["st_rank", "st_z", "st_gap", "inner_st_edge", "inner_pow_edge", "tt_rank", "tt_gap"]


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def add_tenkai(df):
    key = ["date", "jcd", "rno"]
    # ST代理: 展示ST→平均ST→レース平均→全体平均 でフォールバック
    st = df["tenji_st"].copy()
    st = st.fillna(df["avg_st"])
    g0 = df.groupby(key)
    st = st.fillna(g0["tenji_st"].transform("mean")).fillna(st.mean())
    df["_st"] = st
    tt = df["tenji_time"].fillna(g0["tenji_time"].transform("mean")).fillna(df["tenji_time"].mean())
    df["_tt"] = tt
    pw = df["nat_win"].fillna(0.0)
    df["_pw"] = pw

    g = df.groupby(key)
    # レース内の速さ順位・偏差・トップとの差(STは小さいほど速い)
    df["st_rank"] = g["_st"].rank(method="min")
    mu, sd = g["_st"].transform("mean"), g["_st"].transform("std").replace(0, np.nan)
    df["st_z"] = ((st - mu) / sd).fillna(0.0)
    df["st_gap"] = st - g["_st"].transform("min")
    df["tt_rank"] = g["_tt"].rank(method="min")
    df["tt_gap"] = tt - g["_tt"].transform("min")

    # 内の艇(lane小)との相対=展開の芯。lane昇順で"自分より内"の累積を作る
    df = df.sort_values(key + ["lane"]).reset_index(drop=True)
    g2 = df.groupby(key, sort=False)
    n_inner = g2.cumcount().values  # 自分より内の艇数(0..5)
    inner_st_mean = np.where(n_inner > 0, (g2["_st"].cumsum().values - df["_st"].values) / np.clip(n_inner, 1, None), df["_st"].values)
    df["inner_st_edge"] = inner_st_mean - df["_st"].values          # 正=内より速く出る=差し/まくりの芽
    inner_pw_max = g2["_pw"].cummax().shift().values                 # 自分含むcummax→shiftで"内だけ"の最大
    # shiftはグループ跨ぎでNaN=最内艇。最内は内が無いので0扱い
    first = n_inner == 0
    inner_pw_max = np.where(first, df["_pw"].values, inner_pw_max)
    df["inner_pow_edge"] = df["_pw"].values - inner_pw_max           # 正=内の誰より地力上
    return df


def fit_eval(df, extra):
    cols = FEATURES + extra
    base = build_frame(df, impute=False)[FEATURES].reset_index(drop=True)
    X = pd.concat([base, df[TENKAI].reset_index(drop=True)], axis=1)[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    acc, rn, rd = [], 0.0, 0
    for yv in [v for v in sorted(df["yr"].unique()) if v not in SELECTION_YEARS]:
        te = (df["yr"] == yv).to_numpy()
        tr = (df["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        out = df[te].copy()
        out["p"] = m.predict_proba(X[te])[:, 1]
        for _, gg in out.groupby(["date", "jcd", "rno"]):
            gg = gg.sort_values("p", ascending=False)
            top = gg.iloc[0]
            if pd.isna(top["tansho_lane"]):
                continue
            acc.append(int(top["tansho_lane"]) == int(top["lane"]))
            if int(top["lane"]) != 1:
                rd += 1
                if int(top["tansho_lane"]) == int(top["lane"]):
                    rn += float(top["tansho_yen"])
    return np.mean(acc) * 100, (rn / (rd * 100) * 100 if rd else 0)


def main():
    df = load()
    df = add_tenkai(df)
    print("展開特徴(レース内相対) 付与完了\n")
    base = fit_eval(df, [])
    print(f"  ベース(14特徴・各艇絶対)  : 1着{base[0]:.1f}% / 妙味単勝{base[1]:.0f}%")
    plus = fit_eval(df, TENKAI)
    print(f"  +展開(6艇の相互作用/相対): 1着{plus[0]:.1f}% / 妙味単勝{plus[1]:.0f}%")
    print(f"\n  -> 上乗せ: 1着 {plus[0]-base[0]:+.1f}pt / 妙味単勝 {plus[1]-base[1]:+.0f}pt")
    if plus[0] - base[0] >= 0.5 or plus[1] - base[1] >= 5:
        print("  ** 動いた！ 展開(相互作用)that初の壁破り＝個別検証へ。走る姿を見た甲斐あり。")
    else:
        print("  展開(相対)も上乗せ無し＝市場thaが1マールの相互作用も織り込み済で天井、で確定。")
    print("\n※確定払戻・walk-forward・リークなし(展示ST/タイムはレース前情報)。")


if __name__ == "__main__":
    main()
