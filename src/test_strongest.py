"""最強の妙味を特定: 生き残った唯一の勝ち筋(2連単EV割安)を、一番濃い断面で切る。

予測は天井/単勝EVは死。生きてるのは2連単EVミスプライスのみ(316-370%・単調・全年・頑健)。
その最強断面＝EV割安 × 高確信(p0=本命の正規化勝率)のクロス。妙味クリーン年・確定払戻で
グリッド化しROI最強セル(＝最強妙味の定義)を出す。N併記(小セルは過学習注意)。

例: python -u -m src.test_strongest
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import storage
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS

EX = os.path.join(storage.DATA_DIR, "exacta_odds_cache.json")


def collect(df):
    exc = json.load(open(EX, encoding="utf-8")) if os.path.exists(EX) else {}
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["exacta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            key = f"{date}-{jcd}-{rno}"
            ex = exc.get(key)
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            pmap = dict(exacta_probs(p))
            if not (ex and all(ex.get(c) for c in top3)):
                continue
            ev = sum(pmap[c] * ex[c] for c in top3) / 3.0
            win = top["exacta_combo"] in top3
            rows.append({"yr": y, "p0": p[int(top["lane"])], "ev": ev,
                         "ret": float(top["exacta_yen"]) if win else 0.0})
    return rows


def cell(rows, p0lo, evlo):
    sub = [r for r in rows if r["p0"] >= p0lo and r["ev"] > evlo]
    n = len(sub)
    if n < 20:
        return f"{n}(薄)"
    ret = sum(r["ret"] for r in sub)
    roi = ret / (n * 300)
    yb = {}
    for r in sub:
        yb.setdefault(r["yr"], [0.0, 0])
        yb[r["yr"]][0] += r["ret"]
        yb[r["yr"]][1] += 300
    allpos = all(v[0] / v[1] > 1 for v in yb.values())
    return f"{n}/{roi*100:.0f}%{'+' if allpos else ''}"


def main():
    df = load()
    rows = collect(df)
    print(f"妙味・2連単EVあり {len(rows):,}レース（本命≠1号・クリーン年・確定払戻）\n")
    print("  セル = N件/ROI（+=全年100%超）｜行=EV割安>, 列=高確信 本命勝率p0≥\n")
    p0s = [0.0, 0.35, 0.45, 0.50]
    evs = [2.0, 2.5, 3.0, 3.5]
    print("  " + f"{'EV>':<7}" + "".join(f"p0≥{int(p*100):>2}%".rjust(14) for p in p0s))
    best = (None, -1.0)
    for ev in evs:
        line = f"  {ev:<7.1f}"
        for p0 in p0s:
            line += f"{cell(rows, p0, ev):>14}"
            sub = [r for r in rows if r["p0"] >= p0 and r["ev"] > ev]
            if len(sub) >= 40:
                roi = sum(r["ret"] for r in sub) / (len(sub) * 300)
                if roi > best[1]:
                    best = ((ev, p0, len(sub)), roi)
        print(line)
    if best[0]:
        ev, p0, n = best[0]
        print(f"\n  ** 最強妙味 = 2連単EV>{ev} × 本命勝率≥{p0*100:.0f}% : {n}件 ROI {best[1]*100:.0f}%")
        sub = [r for r in rows if r["p0"] >= p0 and r["ev"] > ev]
        rets = np.sort(np.array([r["ret"] for r in sub]))[::-1]
        stake = len(sub) * 300
        print("  fat-tail: " + " ".join(f"{k}本->{rets[k:].sum()/stake*100:.0f}%" for k in [0, 1, 5, 10]))
    print("\n※確定払戻・ライブ未証明(締切29%乖離で実ライブは低め)。小セルは過学習注意。")
    print("  これが5方向×2市場を潰し切って残った、唯一生きてる勝ち筋の最濃断面。")


if __name__ == "__main__":
    main()
