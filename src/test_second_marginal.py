"""#1 検証: 2連単エッジは「1着contrarianエッジ × 非インのオッズ人気バイアス」だけか、
それとも2着マージナル(本命が1着を取った条件下で実2着を当てる力)も効いてるか。

設計(統制実験):
  1着 = モデル本命(≠1号艇) を全戦略で固定。2着の3点選択だけを差し替える。
  - 本命が1着を外したら全戦略0点=差は出ない。
  - 差は「本命が1着を取ったレースで実2着を拾えたか」だけから来る=2着マージナルそのもの。

戦略(2着候補セット S, |S|=3。full5のみ5):
  model_top3 : 残り艇のモデル勝率 上位3   (現行に相当)
  random3    : 残り5艇からランダム3 (多数シードで平均)
  model_bot3 : 残り艇のモデル勝率 下位3   (アンチモデル)
  full5      : 残り5艇 全張り(2着予測せず=1着エッジ純度、5点)

判定:
  model_top3 ≒ random3 ≒ (full5を点数正規化) → 2着選択は無価値 = #1の仮説YES(1着×オッズ構造)
  model_top3 が有意に上 → 2着マージナルが実在 = 3連単を畳んだ理由と矛盾(要精査)

理論: 2着に情報ゼロかつ配当が2着に無相関なら random3_ROI ≒ full5_ROI
  (ランダム3点は各1着的中レースの配当を確率3/5で拾い、点数も3/5→相殺)。

例: python -u -m src.test_second_marginal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validate import load, fit_predict_leakfree, SELECTION_YEARS


def collect(df, clean_only):
    """contrarianレースごとに 本命/残り艇モデル順/実1着-2着/配当 を1件化。"""
    rows = []
    years = sorted(df["yr"].unique())
    for y in years:
        if clean_only and y in SELECTION_YEARS:
            continue
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for _, g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            honmei = int(top["lane"])
            if honmei == 1 or pd.isna(top["exacta_yen"]) or pd.isna(top["exacta_combo"]):
                continue
            # 残り艇をモデル勝率降順で
            rem = [int(l) for l in g["lane"] if int(l) != honmei]  # 既に p 降順
            combo = str(top["exacta_combo"])
            parts = combo.split("-")
            won_1st = len(parts) == 2 and int(parts[0]) == honmei
            actual_2nd = int(parts[1]) if won_1st else None
            rows.append({
                "rem": rem, "won_1st": won_1st, "actual_2nd": actual_2nd,
                "pay": float(top["exacta_yen"]),
            })
    return rows


def roi_fixed(recs, pick_fn, n_pts):
    """pick_fn(rem)->2着セット。ROI=Σ(pay if 本命1着&実2着∈S else 0)/(n_pts*100*R)。"""
    ret = 0.0
    R = len(recs)
    for r in recs:
        if r["won_1st"] and r["actual_2nd"] in pick_fn(r["rem"]):
            ret += r["pay"]
    return ret / (n_pts * 100 * R), R


def roi_random(recs, seed, k=3):
    rng = np.random.default_rng(seed)
    ret = 0.0
    R = len(recs)
    for r in recs:
        S = set(rng.choice(r["rem"], size=min(k, len(r["rem"])), replace=False).tolist())
        if r["won_1st"] and r["actual_2nd"] in S:
            ret += r["pay"]
    return ret / (k * 100 * R)


def report(recs, label):
    R = len(recs)
    won = sum(1 for r in recs if r["won_1st"])
    print("\n" + "#" * 62 + f"\n### {label}  contrarianレース {R:,} / 本命が1着 {won:,} ({won/R*100:.1f}%)")

    top3_roi, _ = roi_fixed(recs, lambda rem: set(rem[:3]), 3)
    bot3_roi, _ = roi_fixed(recs, lambda rem: set(rem[-3:]), 3)
    full5_roi, _ = roi_fixed(recs, lambda rem: set(rem), 5)
    rand = np.array([roi_random(recs, s) for s in range(30)])

    print(f"  {'戦略':<16}{'点数':>4}{'ROI':>9}   {'メモ'}")
    print(f"  {'model_top3':<16}{3:>4}{top3_roi*100:>8.1f}%   モデル2着上位3(現行相当)")
    print(f"  {'random3(平均)':<16}{3:>4}{rand.mean()*100:>8.1f}%   ±{rand.std()*100:.1f}pt (30シード) [{rand.min()*100:.0f},{rand.max()*100:.0f}]")
    print(f"  {'model_bot3':<16}{3:>4}{bot3_roi*100:>8.1f}%   アンチモデル2着")
    print(f"  {'full5':<16}{5:>4}{full5_roi*100:>8.1f}%   2着全張り(1着エッジ純度)")
    print(f"\n  理論チェック: 2着情報ゼロなら random3 ≒ full5 → 実測 {rand.mean()*100:.1f}% vs {full5_roi*100:.1f}%")
    edge = (top3_roi - rand.mean()) / rand.std() if rand.std() > 0 else float("inf")
    print(f"  2着スキル判定: (model_top3 - random3)/σ = {edge:+.1f}σ", end="  ")
    if abs(edge) < 2:
        print("→ 区別つかず = 2着選択は無価値 = #1仮説YES(1着×オッズ構造)")
    elif edge >= 2:
        print("→ model_top3が有意に上 = 2着マージナル実在(要精査)")
    else:
        print("→ model_top3が有意に下 = モデル2着は逆効果")


def main():
    df = load()
    print("2連単エッジの分解: 1着contrarian × オッズ構造 か / 2着マージナルも効くか")
    for clean_only, lab in [(True, "クリーン年のみ(2023-25)"), (False, "全年")]:
        recs = collect(df, clean_only)
        report(recs, lab)


if __name__ == "__main__":
    main()
