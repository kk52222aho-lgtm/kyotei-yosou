"""R3' の天井を測る + 縮/広を後ろ向きに見る。**どっちも事前登録やない。**

=== ① R3' の天井(登録する前に必ず測る) ===
R3 は T-1 のオッズでEVを計算して死んだ(ROI 76%)。死因は「T-1で高いオッズは
まだ金が入ってへんだけ」= 発火艇の維持率 0.532。

ほな**確定オッズを完璧に当てられたら勝てるんか。**それを先に測る。
確定オッズ O_final は録ってある(T+3窓)。**それを使うてEVを組む。**
これは実戦では不可能(締切後にしか分からん)やから、**天井**や。

  **天井が 100% を割ったら、維持率モデルをどれだけ良うしても R3' は成立せん。**
  → その時点でEV系の枝を丸ごと閉じる。事前登録を書く必要すら無い。

=== ② 縮/広(後ろ向き・事前登録やない) ===
`export_task.log` が毎日刷っとる「有効N 392/300」の実験や。**条項本文は
作業ツリー・git全履歴・diff(-S)を全部当たって存在せんことを確認した**
(`export_odds.py` はそもそも一度もコミットされとらん)。

→ **これは事前登録付きの前向き検証やない。後ろ向き分析や。**
  「合格」は名乗れん。用途は**第二封筒の事前分布を作ること**だけ。
  ここで出た数字で条項を書いて、N を 0 から積み直す。

規律: 払戻は確定払戻(payouts.tansho_yen)。CI は日付クラスタ bootstrap。
     母集団全体を必ず並べる。

usage: python -u -m src.test_ev_ceiling
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.default_rng(20260908)
NBOOT = 4000
THR = 1.15          # R3 と同じ閾値。**動かさん**(動かしたら庭になる)


def ci(ret, n_bets, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    r = np.bincount(iv, weights=ret, minlength=len(ud))
    c = np.bincount(iv, weights=n_bets, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    return np.percentile(r[idx].sum(1) / (c[idx].sum(1) * 100), [2.5, 97.5])


def load():
    conn = storage.connect()
    d = pd.read_sql_query("""
        SELECT date, jcd, rno, combo, mins_to_deadline m, odds, model_p,
               honmei, gyaku
        FROM odds_timeseries WHERE bet_type='tansho'""", conn)
    pay = pd.read_sql_query("SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
                            "WHERE tansho_lane IS NOT NULL", conn)
    conn.close()
    for x in (d, pay):
        x["jcd"] = x["jcd"].astype(str).str.zfill(2)
    d["lane"] = pd.to_numeric(d["combo"], errors="coerce")
    k = ["date", "jcd", "rno", "lane"]
    w = d.pivot_table(index=k, columns="m", values="odds")
    w.columns = [f"o{int(c)}" for c in w.columns]
    meta = d[d.m == 1].set_index(k)[["model_p", "honmei", "gyaku"]]
    X = w.join(meta, how="inner").reset_index()
    X = X.merge(pay, on=["date", "jcd", "rno"], how="inner")
    X["won"] = (X["lane"] == X["tansho_lane"]).astype(int)
    return X


def main():
    X = load()
    X = X.dropna(subset=["o1", "model_p"])
    print(f"盤: {len(X):,}艇 / {X.groupby(['date','jcd','rno']).ngroups:,}レース")

    # レース内でモデル確率を正規化(EVの分母を揃える)
    g = X.groupby(["date", "jcd", "rno"])
    X["p"] = X["model_p"] / g["model_p"].transform("sum")
    X["ev_t1"] = X["p"] * X["o1"]
    has_f = X["o-3"].notna()
    X.loc[has_f, "ev_fin"] = X.loc[has_f, "p"] * X.loc[has_f, "o-3"]

    print("\n" + "=" * 74)
    print("■ ① R3' の天井 — 確定オッズでEVを組んだらどうか(実戦では不可能)")
    print("=" * 74)
    sub = X[has_f].copy()
    print(f"  確定オッズが録れとる {len(sub):,}艇 / "
          f"{sub.groupby(['date','jcd','rno']).ngroups:,}レース")
    for lab, col in [("R3 と同じ = T-1 のEV(実測で死んだ側)", "ev_t1"),
                     ("**天井 = 確定オッズのEV**", "ev_fin")]:
        gg = sub.groupby(["date", "jcd", "rno"])
        idx = gg[col].idxmax()
        top = sub.loc[idx]
        top = top[top[col] >= THR]
        if not len(top):
            print(f"  {lab}: 発火 0")
            continue
        ret = np.where(top["won"] == 1, top["tansho_yen"], 0.0)
        roi = ret.sum() / (len(top) * 100)
        lo, hi = ci(ret, np.ones(len(top)), top["date"].to_numpy())
        print(f"\n  {lab}")
        print(f"    発火 {len(top):,}件  的中 {top['won'].mean():.2%}  "
              f"回収 {roi:.1%}  CI[{lo:.1%},{hi:.1%}]")
    # 全艇ベタ(相手)
    allb = sub.groupby(["date", "jcd", "rno"]).head(6)
    ret = np.where(allb["won"] == 1, allb["tansho_yen"], 0.0)
    print(f"\n  相手: 全艇ベタ買い  {len(allb):,}点  "
          f"回収 {ret.sum() / (len(allb) * 100):.1%}")
    b1 = sub[sub["lane"] == 1]
    ret1 = np.where(b1["won"] == 1, b1["tansho_yen"], 0.0)
    print(f"  相手: 1号艇ベタ買い(予測不要)  {len(b1):,}点  "
          f"回収 {ret1.sum() / (len(b1) * 100):.1%}")

    print("\n" + "=" * 74)
    print("■ ② 縮/広(後ろ向き。事前登録やない。『合格』は名乗れん)")
    print("=" * 74)
    fl = X[(X["honmei"] != 1) & (X["lane"] == X["honmei"])].copy()
    fl = fl.dropna(subset=["o10", "o1"])
    fl["move"] = np.log(fl["o1"] / fl["o10"])          # T-10 → T-1 の動き
    print(f"  FLB(本命≠1号)で両端(T-10 & T-1)がそろう本命艇 {len(fl):,}件")
    fl["grp"] = np.where(fl["move"] < 0, "縮(値が下がった)", "広(値が上がった)")
    print(f"\n  {'群':<18}{'件数':>7}{'的中率':>9}{'回収率':>9}{'95%CI':>20}"
          f"{'T-1オッズ中央':>13}")
    for k, s in fl.groupby("grp"):
        ret = np.where(s["won"] == 1, s["tansho_yen"], 0.0)
        roi = ret.sum() / (len(s) * 100)
        lo, hi = ci(ret, np.ones(len(s)), s["date"].to_numpy())
        print(f"  {k:<18}{len(s):>7,}{s['won'].mean():>9.2%}{roi:>9.1%}"
              f"  [{lo:>6.1%},{hi:>6.1%}]{s['o1'].median():>13.2f}")
    ret = np.where(fl["won"] == 1, fl["tansho_yen"], 0.0)
    lo, hi = ci(ret, np.ones(len(fl)), fl["date"].to_numpy())
    print(f"  {'全体(縮広を分けん)':<18}{len(fl):>7,}{fl['won'].mean():>9.2%}"
          f"{ret.sum() / (len(fl) * 100):>9.1%}  [{lo:>6.1%},{hi:>6.1%}]")

    # 動きの大きさで割る
    print(f"\n  動きの大きさ(|log(O_T-1/O_T-10)|)で五分位に割る:")
    fl["q"] = pd.qcut(fl["move"], 5, labels=False, duplicates="drop")
    print(f"  {'五分位':<8}{'件数':>7}{'move中央':>10}{'的中率':>9}{'回収率':>9}")
    for k, s in fl.groupby("q"):
        ret = np.where(s["won"] == 1, s["tansho_yen"], 0.0)
        print(f"  {int(k) + 1:<8}{len(s):>7,}{s['move'].median():>10.3f}"
              f"{s['won'].mean():>9.2%}{ret.sum() / (len(s) * 100):>9.1%}")
    print("  ※ 単勝の払戻率は75%(控除25%)。金になるのは100%を超えた時だけ。2026-09-08訂正: 旧記載85%は誤り(実測下界73.5%)")
    print("    2026-09-08訂正: それまで85%と書いとったんは誤り(実測下界73.5%)")

if __name__ == "__main__":
    main()
