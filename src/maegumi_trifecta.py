"""1号艇が飛ぶと分かっとる8%のレースで、3連単が買えるか。

`maegumi_detector` で 19,851レース(全体の8%)を特定した。そこでは
1号艇勝率が 54.89% → 38.67%(-16.2pt)。**やが単勝は一番マシな2号艇で89.1%**、
控除の壁に届かん。市場も前づけを見とる。

3連単は配当の桁が違う。頭が割れるレースなら、単勝と質が違う可能性がある。

規律:
- **確定払戻のみ**(trifecta_yen)。オッズ推定は使わん
- 買い目は**先に決めて動かさん**。当たった後で点数を足さん
- **相手は必ず出す**: 同じレースで「1-2-3」を機械的に買う手、全レースで同じ買い方をする手
- CI は日付クラスタのブートストラップ
- walk-forward の p_w1(その年より前だけで学習)で選んだレースのみ

usage: python -u -m src.maegumi_trifecta
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.default_rng(20260902)
NBOOT = 2000


def ci(win, stake, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    w = np.bincount(iv, weights=win, minlength=len(ud))
    s = np.bincount(iv, weights=stake, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    return np.percentile(w[idx].sum(1) / s[idx].sum(1), [2.5, 97.5])


def report(label, df, picks_col, yen_col="trifecta_yen"):
    """picks_col = 各レースの買い目リスト。100円均等。"""
    n_bets = df[picks_col].apply(len)
    hit = [row[yen_col] if row["trifecta_combo"] in row[picks_col] else 0.0
           for _, row in df.iterrows()]
    hit = np.array(hit, float)
    stake = n_bets.to_numpy(float) * 100
    roi = hit.sum() / stake.sum()
    lo, hi = ci(hit, stake, df["date"].to_numpy())
    nh = int((hit > 0).sum())
    print(f"  {label:<34}{len(df):>7,}R {int(n_bets.sum()):>8,}点 "
          f"的中{nh:>5,}({nh / len(df):>6.2%})  回収 {roi:>6.1%} "
          f"CI[{lo:.1%},{hi:.1%}]")
    return roi


def main():
    conn = storage.connect()
    pay = pd.read_sql_query("SELECT date,jcd,rno,trifecta_combo,trifecta_yen "
                            "FROM payouts WHERE trifecta_yen IS NOT NULL", conn)
    conn.close()
    pay["jcd"] = pay["jcd"].astype(str).str.zfill(2)
    R = pd.read_csv("data/maegumi_races.csv", dtype={"date": str, "jcd": str})
    R["jcd"] = R["jcd"].str.zfill(2)
    R = R.dropna(subset=["p_w1"]).merge(pay, on=["date", "jcd", "rno"], how="inner")
    R["dec"] = pd.qcut(R["p_w1"], 10, labels=False, duplicates="drop")
    print(f"盤: {len(R):,}レース(walk-forwardのp_w1が付いた確定払戻レース)")
    print(f"   最下位十分位 = 1号艇が危ない: {(R.dec == 0).sum():,}R  "
          f"1号艇勝率 {R.loc[R.dec == 0, 'win1'].mean():.2%} "
          f"(全体 {R['win1'].mean():.2%})\n")

    danger = R[R["dec"] == 0].copy()
    rest = R[R["dec"] > 0].copy()

    # ---- 買い目を先に定義(結果を見て動かさん) ----
    def fmt(t):
        return f"{t[0]}-{t[1]}-{t[2]}"

    heads23 = [fmt(t) for t in itertools.permutations([1, 2, 3, 4, 5, 6], 3)
               if t[0] in (2, 3)]
    sets = {
        "頭2,3 × 相手全通り(40点)": heads23,
        "頭2,3 相手1,3,4/1,2,4 に絞る(12点)":
            [fmt((h, a, b)) for h in (2, 3)
             for a, b in itertools.permutations([x for x in (1, 2, 3, 4)
                                                 if x != h], 2)],
        "2-1-3 / 3-1-2 の2点": ["2-1-3", "3-1-2"],
        "頭2 の6点": [fmt((2, a, b)) for a, b in
                     itertools.permutations([1, 3, 4, 5, 6], 2)][:6],
        "1-2-3 の1点(相手)": ["1-2-3"],
        "頭1 × 相手全通り(20点)(相手)": [fmt(t) for t in
                                  itertools.permutations([1, 2, 3, 4, 5, 6], 3)
                                  if t[0] == 1],
    }

    print("=== ① 1号艇が飛ぶ8%のレース(n={:,}) ===".format(len(danger)))
    for k, v in sets.items():
        danger["_p"] = [v] * len(danger)
        report(k, danger, "_p")

    print(f"\n=== ② 残り92%のレース(同じ買い目・対照) n={len(rest):,} ===")
    for k in ["頭2,3 × 相手全通り(40点)", "1-2-3 の1点(相手)",
              "頭1 × 相手全通り(20点)(相手)"]:
        rest["_p"] = [sets[k]] * len(rest)
        report(k, rest, "_p")

    print(f"\n=== ③ 危険度を細かく割る(頭2,3の40点で固定) ===")
    R["_p"] = [sets["頭2,3 × 相手全通り(40点)"]] * len(R)
    print(f"  {'十分位':<8}{'レース':>8}{'1号艇勝率':>10}{'回収率':>9}")
    for k, g in R.groupby("dec"):
        hit = np.array([r["trifecta_yen"] if r["trifecta_combo"] in r["_p"] else 0.0
                        for _, r in g.iterrows()], float)
        roi = hit.sum() / (len(g) * 40 * 100)
        print(f"  {int(k) + 1:<8}{len(g):>8,}{g['win1'].mean():>10.2%}{roi:>9.1%}")

    print(f"\n=== ④ 前半/後半で割る(頭2,3の40点・危険8%のみ) ===")
    for lab, sub in [("前半 2022-2024", danger[danger.date < "20250101"]),
                     ("後半 2025-2026", danger[danger.date >= "20250101"])]:
        sub = sub.copy()
        sub["_p"] = [sets["頭2,3 × 相手全通り(40点)"]] * len(sub)
        report(lab, sub, "_p")


if __name__ == "__main__":
    main()
