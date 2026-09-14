"""走査の器そのものを偽薬で点呼する。912/1320 が生き残ったのを信じる前に。

走査(scan_all.py)が「増分で閾値超え 912/1320」を出した。多すぎる。
器が壊れとって t を吹かしとる可能性を先に潰す。**同じ配管に、答えを知っとる列を流す。**

  ① 純ノイズ          → t は N(0,1) のはず。閾値(3.93)を超えたらアウト
  ② 強い特徴のレース入替 → レース塊ごと並べ替えて関係を切る。t は N(0,1) のはず
  ③ 今レースの艇番      → 既存モデルに入っとる。増分 t ≈ 0 のはず
  ④ 今レースの全国勝率   → 既存モデルに入っとる。増分 t ≈ 0 のはず
  ⑤ 今レースの展示タイム → 既存モデルに入っとる。増分 t ≈ 0 のはず
  ⑥ 今レースの着順      → **答えそのもの。**生 t が桁違いに出んかったら配管が死んどる

③④⑤が 0 に落ちて⑥が爆発したら、器は「既に入っとるもんは 0、入っとらんもんは検出」を
できとる。そのとき初めて 912 を読み始める。

usage: python -u -m src.scan_placebo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scan_all import KEYS, lagged, load, race_z

RNG = np.random.default_rng(20260903)


def main():
    df = load()
    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    df = df.merge(oof, on=["date", "jcd", "rno", "lane"], how="left")
    rid_codes, uniq = pd.factorize(df["rid"], sort=False)
    nrace = len(uniq)
    assert len(df) == nrace * 6

    p = df["p"].to_numpy(dtype=float)
    win = df["win"].to_numpy(dtype=float)
    have = ~np.isnan(p) & ~np.isnan(win)
    keep = np.bincount(rid_codes, weights=have.astype(float), minlength=nrace)[rid_codes] == 6
    psum = np.bincount(rid_codes, weights=np.where(keep, p, 0.0), minlength=nrace)
    q = np.where(keep, p / np.maximum(psum[rid_codes], 1e-12), np.nan)
    resid = np.where(keep, win - q, np.nan)
    raw_r = np.where(keep, win - 1 / 6, np.nan)
    dcodes, dun = pd.factorize(df["dnum"], sort=True)
    nd = len(dun)
    print(f"レース {nrace:,} / 採点対象 {int(keep.sum() // 6):,} / 日 {nd:,}")

    def stat(vals, label):
        z = race_z(vals, nrace)
        m = keep & ~np.isnan(z)
        n = int(m.sum() // 6)
        if n < 100:
            print(f"  {label:34s} 使えるレースが足らん({n})")
            return
        zi = np.where(m, z, 0.0)
        out = []
        for r in (resid, raw_r):
            W = np.bincount(dcodes, weights=zi * np.where(m, r, 0.0), minlength=nd)
            den = np.sqrt((W ** 2).sum())
            out.append(W.sum() / den if den else np.nan)
        print(f"  {label:34s} 増分t={out[0]:+9.2f}   生t={out[1]:+9.2f}   レース {n:,}")

    print("\n--- ① 純ノイズ(閾値3.93を超えたら器が壊れとる) ---")
    for i in range(5):
        stat(RNG.random(len(df)), f"乱数 #{i + 1}")

    print("\n--- ② 強い特徴をレース塊ごと入替(関係を切っただけ) ---")
    strong = None
    for lname, vals in lagged(df, KEYS["選手"], "win"):
        if lname == "wall":
            strong = vals
            break
    stat(strong, "選手|win|wall (そのまま)")
    for i in range(3):
        perm = RNG.permutation(nrace)
        stat(strong.reshape(nrace, 6)[perm].ravel(), f"同上・レース入替 #{i + 1}")

    print("\n--- ③〜⑤ 既に既存モデルに入っとる量(増分は0のはず) ---")
    stat(df["lane"].to_numpy(float), "今レースの艇番 lane")
    stat(df["natwin"].to_numpy(float), "今レースの全国勝率 nat_win")
    stat(pd.to_numeric(df["tenji_time"], errors="coerce").to_numpy(float),
         "今レースの展示タイム")

    print("\n--- ⑥ 答えそのもの(配管の死活) ---")
    stat(-df["fin"].to_numpy(float), "今レースの着順(符号反転)")


if __name__ == "__main__":
    main()
