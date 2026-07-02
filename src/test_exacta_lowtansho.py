"""単勝が低い(大本命)レースで、2連単3点は効くか？（クリーン年 walk-forward）

問い: 単勝オッズ<1.5(本命がほぼ1着確実)のレースでは、2連単3点も死に金か？
  = 前に棚上げした「単勝鉄板レースの2連単」に一歩踏み込む。

手順:
  1. クリーン年(2023-25)で past-only 学習 → 妙味レース(本命≠1号艇)
  2. 各レースの本命の確定単勝オッズを取得 → 単勝<1.5 と >=1.5 に分割
  3. 各群で 2連単3点(確率上位3) の回収率・的中数を比較（払戻はデータ既有）

規律: 的中数を先に見る。群の的中数が二桁未満なら「サンプル不足・判定保留」。

例:
  python -m src.test_exacta_lowtansho --sample 150
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import scraper
from .strategy_search import exacta_probs
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
            if int(top["lane"]) == 1 or pd.isna(top["exacta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            rows.append({
                "date": date, "jcd": jcd, "rno": int(rno),
                "honmei": int(top["lane"]),
                "top3": top3,
                "win_combo": top["exacta_combo"],
                "yen": top["exacta_yen"],
                "hit": top["exacta_combo"] in top3,
            })
        if len(rows) > sample_per_year:
            idx = np.linspace(0, len(rows) - 1, sample_per_year).astype(int)
            rows = [rows[i] for i in idx]
        bets.extend(rows)
    return bets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150)
    args = ap.parse_args()

    bets = collect(args.sample)
    print(f"対象 {len(bets)} レース。本命の確定単勝オッズを取得中…（1件約1秒）")
    got = []
    for i, b in enumerate(bets):
        od = scraper.fetch_odds(b["date"], b["jcd"], b["rno"])
        o = od.get(b["honmei"]) if od else None
        if o:
            b["tansho_odds"] = o
            got.append(b)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(bets)}")

    d = pd.DataFrame(got)
    print(f"\nオッズ取得成功 {len(d)}レース")

    def evalset(sub):
        races = len(sub)
        hits = int(sub["hit"].sum())
        stake = races * 3 * 100
        ret = int(sub.loc[sub["hit"], "yen"].sum())
        roi = ret / stake if stake else float("nan")
        return races, hits, roi

    low = d[d["tansho_odds"] < 1.5]
    high = d[d["tansho_odds"] >= 1.5]

    print("\n=== 単勝オッズ帯別 2連単3点 の成績 ===")
    print(f"{'群':<18}{'レース':>7}{'的中数':>7}{'回収率':>9}")
    for label, sub in [("単勝<1.5(大本命)", low), ("単勝>=1.5", high), ("全体", d)]:
        r, h, roi = evalset(sub)
        print(f"{label:<18}{r:>7}{h:>7}{roi*100:>8.1f}%")

    print("\n--- 判定（まず的中数）---")
    r_l, h_l, roi_l = evalset(low)
    r_h, h_h, roi_h = evalset(high)
    if h_l < 10:
        print(f"⚠ 単勝<1.5群の的中数 {h_l}（二桁未満）＝サンプル不足・判定保留。")
        print("  この群の2連単が効く/効かんは、現データでは結論保留。今の『2連単はそのまま』を維持。")
    elif roi_l < 100:
        print(f"単勝<1.5群の2連単は回収率{roi_l*100:.0f}%（的中{h_l}本）＝低調。"
              "『単勝が低いレースは2連単も見送り』が有効の可能性。")
    else:
        print(f"単勝<1.5群でも2連単は回収率{roi_l*100:.0f}%（的中{h_l}本）＝効いてる。見送り不要。")


if __name__ == "__main__":
    main()
