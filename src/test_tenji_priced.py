"""展示の進入隊形は、締切までにオッズへ入るか。**市場が遅れとる唯一の候補。**

=== なぜここか ===
今日ここまでで「モデルは正しいが値段が知っとる」が3回出た(走査・押し出し検出器・FLB)。
市場が効率的やないとしたら、**市場が取りにくい情報**の所しか無い。

展示のスタート展示表は:
  - **締切15分前**に出る(beforeinfo の取得中央値 = T-15)
  - **表が1行に潰れる形式**で、今日パーサを直すまで構造化列は0行やった
  - その**並び順が本番の進入コースと 95.8% 一致**する(n=6,606)
  - そして進入が崩れると1号艇の勝率は 56.8% → 46.8%、
    1号艇自身が2コース以下に落ちたら **11.75%** まで落ちる

**これだけの情報が、15分でオッズに入るんか。**入っとったら②は本当に終い。
入っとらんかったら、そこが唯一の穴や。

=== 測り方(数字を見る前に決めた) ===
(1) レース内で正規化した implied prob(1/オッズ を race 内で和1に)と**実際の勝率**を並べる
(2) 窓ごとに出す: T-15(展示が出た直後) / T-1 / 確定。**入っていく過程が見える**
(3) 群は展示の隊形で切る: 枠なり / 前づけあり / **1号艇が2コース以下**
(4) 相手は必ず置く: 同じ窓の全レース平均
(5) ROI は確定払戻。CI は日付クラスタ bootstrap

**implied prob > 実勝率 なら過剰人気=買うたら損。逆なら穴。**

usage: python -u -m src.test_tenji_priced
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.default_rng(20260908)
NBOOT = 4000
WINDOWS = [15, 10, 5, 1, -3]


def ci(ret, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    s = np.bincount(iv, weights=ret, minlength=len(ud))
    c = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    return np.percentile(s[idx].sum(1) / (c[idx].sum(1) * 100), [2.5, 97.5])


def main():
    conn = storage.connect()
    t = pd.read_sql_query("SELECT date,jcd,rno,lane,tenji_course FROM tenji_st", conn)
    o = pd.read_sql_query("SELECT date,jcd,rno,combo,mins_to_deadline m,odds "
                          "FROM odds_timeseries WHERE bet_type='tansho'", conn)
    p = pd.read_sql_query("SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
                          "WHERE tansho_lane IS NOT NULL", conn)
    conn.close()
    for x in (t, o, p):
        x["jcd"] = x["jcd"].astype(str).str.zfill(2)
    o["lane"] = pd.to_numeric(o["combo"], errors="coerce")
    k = ["date", "jcd", "rno", "lane"]
    w = o.pivot_table(index=k, columns="m", values="odds")
    w.columns = [f"o{int(c)}" for c in w.columns]
    X = t.merge(w.reset_index(), on=k, how="inner").merge(
        p, on=["date", "jcd", "rno"], how="inner")
    X["won"] = (X["lane"] == X["tansho_lane"]).astype(int)
    rid = ["date", "jcd", "rno"]
    n6 = X.groupby(rid)["lane"].transform("size")
    X = X[n6 == 6].copy()
    print(f"盤: {X.groupby(rid).ngroups:,}レース "
          f"({X['date'].min()}-{X['date'].max()})  展示の隊形とオッズが両方ある")

    # レース単位の群
    R = X.groupby(rid).apply(lambda g: pd.Series({
        "c1": int(g.loc[g.lane == 1, "tenji_course"].iloc[0]),
        "mg": int((g.tenji_course != g.lane).any()),
    }), include_groups=False).reset_index()
    X = X.merge(R, on=rid)
    X["grp"] = np.where(X["c1"] > 1, "展示で1号艇が押し出し",
                        np.where(X["mg"] == 1, "展示で前づけあり", "展示は枠なり"))

    b1 = X[X["lane"] == 1].copy()
    print(f"\n=== ① 1号艇: 市場のimplied prob vs 実勝率 ===")
    print(f"  {'群':<22}{'レース':>7}{'実勝率':>9}"
          + "".join(f"{'T-' + str(v) if v > 0 else '確定':>9}" for v in WINDOWS))
    for g, s in b1.groupby("grp"):
        # レース内で 1/odds を和1に正規化
        line = f"  {g:<22}{len(s):>7,}{s['won'].mean():>9.2%}"
        for v in WINDOWS:
            col = f"o{v}"
            if col not in X:
                line += f"{'—':>9}"
                continue
            sub = X.dropna(subset=[col])
            sub = sub[sub.groupby(rid)[col].transform("size") == 6]
            sub = sub.assign(imp=1 / sub[col])
            sub["q"] = sub["imp"] / sub.groupby(rid)["imp"].transform("sum")
            v1 = sub[(sub.lane == 1) & (sub.grp == g)]
            line += f"{v1['q'].mean():>9.2%}" if len(v1) > 30 else f"{'—':>9}"
        print(line)
    print("  ※ implied prob > 実勝率 = 過剰人気(買うたら損) / < = 穴")

    print(f"\n=== ② 1号艇 単勝を買ったらどうなるか(確定払戻) ===")
    print(f"  {'群':<22}{'レース':>7}{'的中率':>9}{'回収率':>9}{'95%CI':>20}")
    for g, s in b1.groupby("grp"):
        ret = np.where(s["won"] == 1, s["tansho_yen"], 0.0)
        lo, hi = ci(ret, s["date"].to_numpy())
        print(f"  {g:<22}{len(s):>7,}{s['won'].mean():>9.2%}"
              f"{ret.sum() / (len(s) * 100):>9.1%}  [{lo:>6.1%},{hi:>6.1%}]")
    ret = np.where(b1["won"] == 1, b1["tansho_yen"], 0.0)
    lo, hi = ci(ret, b1["date"].to_numpy())
    print(f"  {'全体(相手)':<22}{len(b1):>7,}{b1['won'].mean():>9.2%}"
          f"{ret.sum() / (len(b1) * 100):>9.1%}  [{lo:>6.1%},{hi:>6.1%}]")

    print(f"\n=== ③ 展示が出てからオッズがどれだけ動いたか(1号艇) ===")
    print(f"  {'群':<22}{'T-15→T-1':>12}{'T-15→確定':>12}{'T-1→確定':>12}")
    for g, s in b1.groupby("grp"):
        s = s.dropna(subset=["o15", "o1", "o-3"])
        if len(s) < 30:
            print(f"  {g:<22}{'(n<30)':>12}")
            continue
        print(f"  {g:<22}{np.log(s['o1'] / s['o15']).median():>12.3f}"
              f"{np.log(s['o-3'] / s['o15']).median():>12.3f}"
              f"{np.log(s['o-3'] / s['o1']).median():>12.3f}")
    print("  ※ 正 = オッズが上がった(人気が落ちた)。展示の悪い情報が入っとるなら正のはず")

    print(f"\n=== ④ 押し出し群で1号艇を切って他を買う ===")
    bad = X[(X["grp"] == "展示で1号艇が押し出し") & (X["lane"] != 1)]
    if len(bad) > 60:
        for L in (2, 3, 4):
            s = bad[bad["lane"] == L]
            if len(s) < 30:
                continue
            ret = np.where(s["won"] == 1, s["tansho_yen"], 0.0)
            lo, hi = ci(ret, s["date"].to_numpy())
            print(f"  {L}号艇単勝 {len(s):>5,}件  的中 {s['won'].mean():>6.2%}  "
                  f"回収 {ret.sum() / (len(s) * 100):>6.1%}  [{lo:>6.1%},{hi:>6.1%}]")
    else:
        print(f"  n={len(bad) // 5 if len(bad) else 0} レースしか無い。判定せん")
    print("  ※ 単勝の払戻率は75%(2026-09-08訂正)。金になる閾値は100%")


if __name__ == "__main__":
    main()
