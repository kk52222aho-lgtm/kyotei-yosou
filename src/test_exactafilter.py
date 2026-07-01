"""2連単レース見送りフィルタの検証（クリーン年 walk-forward）。

前処理:
  - 2連単の実効控除率を過去データから逆算: 各レース takeout = 1 - 1/Σ(1/オッズ)
  - 損益分岐オッズ = 1/(1-実効控除率) → 見送り閾値（固定・総当たりしない）
検証:
  - 妙味レースの本命2連単オッズ(確率1位の組番)が閾値未満 → そのレースの2連単を見送り
  - 単勝フィルタとは独立
出力:
  - 見送り後の残レース数・的中数・回収率（見送り版 vs 現状）
  - 的中数が二桁未満なら「サンプル不足・判定保留」

例:
  python -m src.test_exactafilter --sample 120
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
            ranked = [c for c, _ in exacta_probs(p)]
            rows.append({
                "date": date, "jcd": jcd, "rno": int(rno),
                "honmei_combo": ranked[0],          # 確率1位の2連単組番
                "top3": ranked[:3],
                "win_combo": top["exacta_combo"],
                "yen": top["exacta_yen"],
            })
        if len(rows) > sample_per_year:
            idx = np.linspace(0, len(rows) - 1, sample_per_year).astype(int)
            rows = [rows[i] for i in idx]
        bets.extend(rows)
    return bets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    args = ap.parse_args()

    bets = collect(args.sample)
    print(f"対象 {len(bets)} レース。2連単オッズ(30組番)を取得中…（1件約1秒）")

    got = []
    for i, b in enumerate(bets):
        od = scraper.fetch_exacta_odds(b["date"], b["jcd"], b["rno"])
        if not od:
            continue
        ho = od.get(b["honmei_combo"])
        if not ho:
            continue
        b["honmei_odds"] = ho
        b["sum_inv"] = sum(1.0 / v for v in od.values())
        b["hit"] = b["win_combo"] in b["top3"]
        got.append(b)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(bets)} 取得")

    d = pd.DataFrame(got)
    if len(d) == 0:
        print("オッズ取得できず。中止。")
        return

    # --- 前処理: 実効控除率 → 損益分岐オッズ ---
    takeout = float((1 - 1 / d["sum_inv"]).mean())
    breakeven = 1.0 / (1 - takeout)
    print(f"\n[前処理] 2連単 実効控除率 = {takeout:.1%} / 損益分岐オッズ = {breakeven:.3f}（＝見送り閾値・固定）")
    print(f"取得成功 {len(d)}レース / 本命組番オッズ 中央値 {d['honmei_odds'].median():.1f} / "
          f"閾値未満は {100*(d['honmei_odds'] < breakeven).mean():.1f}%")

    def evalset(sub, label):
        n = len(sub)
        hits = int(sub["hit"].sum())
        stake = n * 3 * 100                     # 2連単3点
        ret = int(sub.loc[sub["hit"], "yen"].sum())
        roi = ret / stake if stake else float("nan")
        return n, hits, roi

    n_all, h_all, roi_all = evalset(d, "現状")
    kept = d[d["honmei_odds"] >= breakeven]
    n_k, h_k, roi_k = evalset(kept, "見送り後")
    removed = n_all - n_k

    print("\n=== 2連単 見送りフィルタ 結果 ===")
    print(f"{'':<12}{'残レース':>8}{'的中数':>7}{'回収率':>9}")
    print(f"{'現状(全部)':<12}{n_all:>8}{h_all:>7}{roi_all*100:>8.1f}%")
    print(f"{'見送り後':<12}{n_k:>8}{h_k:>7}{roi_k*100:>8.1f}%   (見送り {removed}レース)")

    print("\n--- 判定（まず的中数、次に見送り件数の順で読む）---")
    if h_k < 10:
        print(f"⚠ 見送り後の的中数 {h_k}（二桁未満）＝サンプル不足・判定保留。回収率が良くても採用しない。")
    elif removed == 0:
        print("見送り0件＝(b)レース見送り方式では低オッズに発火せず。")
        print("  本命≠1号艇の妙味レースでは本命2連単組番が構造的に高オッズになるため。")
        print("  ※これは『2連単に低オッズ問題なし』ではない。買う3点内の低オッズ組番(死に金)は")
        print("    (a)個別組番間引き方式の別テーマ＝本テスト対象外・棚上げ。")
        print("  結論: (b)方式では効果なし＝2連単は触らずシンプルに保つ、で正しく閉じる。")
    elif roi_k > roi_all:
        print(f"(b)見送りで回収率 {roi_all*100:.1f}%→{roi_k*100:.1f}%（的中{h_k}本）。有効の可能性。")
        print("  ※あくまで(b)レース見送りの効果。3点内の死に金問題(a)は別テーマ。")
    else:
        print(f"(b)見送りで回収率が上がらない（{roi_all*100:.1f}%→{roi_k*100:.1f}%）＝(b)方式では不要。")


if __name__ == "__main__":
    main()
