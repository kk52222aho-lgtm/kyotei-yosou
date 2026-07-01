"""単勝オッズ下限フィルタの検証（クリーン年・前進検証）。

問い: 「本命の単勝オッズが低すぎる(例1.1倍)レースを見送ると回収率は上がるか？」
控除率25%の壁で低オッズは数学的に不利（1.1倍は勝率91%超でないと+EVにならない）。
これを実データで確かめる。

手順:
  1. クリーン年(戦略未使用の2023-25)で past-only 学習→妙味レース(本命≠1号艇)を抽出
  2. 各レースの本命の確定単勝オッズを oddstf から取得（勝ち負け問わず必要）
  3. オッズ下限 X でフィルタしたときの回収率を比較

例:
  python -m src.test_oddsfilter --sample 150
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import scraper
from .validate import load, fit_predict_leakfree, SELECTION_YEARS


def collect(sample_per_year: int):
    df = load()
    bets = []
    for y in sorted(df["yr"].unique()):
        if y in SELECTION_YEARS:
            continue
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        rows = []
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["tansho_lane"]):
                continue
            rows.append({
                "date": date, "jcd": jcd, "rno": int(rno),
                "honmei": int(top["lane"]),
                "won": int(top["tansho_lane"]) == int(top["lane"]),
                "tansho_yen": top["tansho_yen"],
            })
        # サンプリング（均等）
        if len(rows) > sample_per_year:
            idx = np.linspace(0, len(rows) - 1, sample_per_year).astype(int)
            rows = [rows[i] for i in idx]
        bets.extend(rows)
    return bets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150, help="1年あたりサンプル数")
    args = ap.parse_args()

    bets = collect(args.sample)
    print(f"対象 {len(bets)} ベット。本命の確定単勝オッズを取得中…（1件約1秒）")

    got = []
    for i, b in enumerate(bets):
        od = scraper.fetch_odds(b["date"], b["jcd"], b["rno"])
        o = od.get(b["honmei"]) if od else None
        if o:
            b["odds"] = o
            got.append(b)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(bets)} 取得")

    df = pd.DataFrame(got)
    print(f"\nオッズ取得成功 {len(df)} ベット / 的中率 {df.won.mean():.1%} / 平均オッズ {df.odds.mean():.2f}")
    print(f"オッズ分布: <1.5倍={100*(df.odds<1.5).mean():.0f}% / 1.5-3={100*((df.odds>=1.5)&(df.odds<3)).mean():.0f}% / 3倍+={100*(df.odds>=3).mean():.0f}%")

    def roi_for(min_odds):
        s = df[df.odds >= min_odds]
        if len(s) == 0:
            return None
        ret = s.loc[s.won, "tansho_yen"].sum()
        return len(s), ret / (len(s) * 100), s.won.mean()

    print("\n=== 単勝オッズ下限フィルタ別 回収率 ===")
    print(f"{'下限':<8}{'ベット':>7}{'的中率':>8}{'回収率':>9}")
    for X in [1.0, 1.3, 1.5, 2.0, 2.5, 3.0]:
        r = roi_for(X)
        if r:
            n, roi, hit = r
            tag = "  ←全部" if X == 1.0 else ""
            print(f"{X:<8}{n:>7}{hit*100:>7.1f}%{roi*100:>8.1f}%{tag}")
    print("\n※控除率25%下で低オッズは不利。下限を上げて回収率が上がれば『低オッズ見送り』が有効。")


if __name__ == "__main__":
    main()
