"""予想精度の改善は、到達率を動かすか。**今日一日の答え合わせ。**

到達率 = (実測ROI − 払戻率)/(1 − 払戻率)。0%=無作為、100%=損益分岐。
1号艇固定(予測が要らん手)は 253,786レースで ROI 90.6% = **到達率 62%**。

今日やったこと:
  - 総当たり走査で本命的中を +0.28pt(holdout)
  - 押し出し検出器で「1号艇が危ないレース」を33.6倍に濃縮
  - どっちも**的中率**の話やった。**ROIは一度も測っとらん**

歴史オッズが無いので253kレースでは測れん。**やが前向き収集の36日ぶんがある。**
同じレース集合の上で、単勝の実測ROIを並べる。

並べるもん(全部 同じレース・同じ確定払戻):
  A **1号艇固定**            ← 予測が要らん手
  B モデル本命(既存14特徴)
  C **押し出し検出器が発火したレースだけ、1号艇を切って2号艇**

問い: **B や C の到達率が A を超えるか。**
超えんかったら、今日の精度改善は的中率だけの話で金には触れてへん。

usage: python -u -m src.test_reach_by_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.default_rng(20260908)
NBOOT = 4000
PAYBACK = 0.75          # 単勝の払戻率(実測下界73.5%・リポジトリ記載 控除25%)


def reach(roi):
    return (roi - PAYBACK) / (1 - PAYBACK)


def ci(ret, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    s = np.bincount(iv, weights=ret, minlength=len(ud))
    c = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    return np.percentile(s[idx].sum(1) / (c[idx].sum(1) * 100), [2.5, 97.5])


def main():
    conn = storage.connect()
    pay = pd.read_sql_query("SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
                            "WHERE tansho_lane IS NOT NULL", conn)
    dates = pd.read_sql_query("SELECT DISTINCT date FROM odds_timeseries", conn)
    conn.close()
    pay["jcd"] = pay["jcd"].astype(str).str.zfill(2)
    keep = set(dates["date"])
    pay = pay[pay["date"].isin(keep)]

    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    oof = oof[oof["date"].isin(keep)]
    top = oof.loc[oof.groupby(["date", "jcd", "rno"])["p"].idxmax(),
                  ["date", "jcd", "rno", "lane"]].rename(columns={"lane": "model_top"})

    # boat1_pushed.csv は rid(=date+jcd+rno2桁)を持っとって rno 列が無い
    push = pd.read_csv("data/boat1_pushed.csv", dtype={"date": str, "jcd": str,
                                                       "rid": str})
    push["jcd"] = push["jcd"].str.zfill(2)
    push = push.dropna(subset=["p"]).copy()
    push["rno"] = push["rid"].str[-2:].astype(int)
    push = push[["date", "jcd", "rno", "p"]].rename(columns={"p": "p_push"})

    d = pay.merge(top, on=["date", "jcd", "rno"], how="inner")
    d = d.merge(push, on=["date", "jcd", "rno"], how="left")
    print(f"盤: {len(d):,}レース ({d['date'].min()}-{d['date'].max()})  "
          f"前向きオッズ収集と重なる期間")
    print(f"払戻率の仮定 {PAYBACK:.0%} / 到達率 = (ROI−{PAYBACK:.0%})/(1−{PAYBACK:.0%})\n")

    thr = float(np.nanquantile(push["p_push"], 0.99))

    def line(lab, sub, pick):
        if len(sub) < 50:
            print(f"  {lab:<34}{len(sub):>6,}R  (n不足)")
            return
        hit = sub["tansho_lane"] == pick
        ret = np.where(hit, sub["tansho_yen"], 0.0)
        roi = ret.sum() / (len(sub) * 100)
        lo, hi = ci(ret, sub["date"].to_numpy())
        print(f"  {lab:<34}{len(sub):>6,}R  的中 {hit.mean():>6.2%}  "
              f"回収 {roi:>6.1%} [{lo:>5.1%},{hi:>6.1%}]  "
              f"**到達率 {reach(roi):>5.0%}**")

    print("=== 全レース ===")
    line("A 1号艇固定(予測が要らん手)", d, 1)
    line("B モデル本命(既存14特徴)", d, d["model_top"])
    print("\n=== モデルが1号艇を切ったレースだけ ===")
    dev = d[d["model_top"] != 1]
    print(f"  (全体の {len(dev) / len(d):.1%})")
    line("A 1号艇固定", dev, 1)
    line("B モデル本命(=1号艇やない)", dev, dev["model_top"])

    print(f"\n=== 押し出し検出器が発火(上位1%, p>={thr:.4f}) ===")
    fire = d[d["p_push"] >= thr]
    print(f"  (全体の {len(fire) / len(d):.1%})")
    line("A 1号艇固定", fire, 1)
    for L in (2, 3):
        line(f"C {L}号艇単勝", fire, L)
    print(f"\n  ※ 参考: 253,786レースでの1号艇固定は ROI 90.6% = 到達率 62%")
    print("  ※ 到達率100%で初めて損益分岐。CIが広いので点推定は参考値")


if __name__ == "__main__":
    main()
