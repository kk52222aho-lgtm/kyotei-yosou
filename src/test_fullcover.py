"""全買い(総流し)の市場効率：賭式ごとに「全組を1点ずつ買った時の回収率」の分布。

問い(step1): 各賭式で、全買い回収率 = 勝ち組払戻 /(組数×100) の分布はどうか。
  - 平均 = 実効払戻率(=1-控除率)。全買いは平均では必ず控除率ぶん負ける。
  - 問いは分布の上側: 100%超えるレースが何本あるか＝市場が理論以上に壊れたレース。
  - 賭式間比較は控除率差を消すため「回収率 / その式の平均回収率」= 相対効率で見る。
規律:
  - 確定払戻(Kファイル)=絶対値。オッズ不要＝締切/確定の曖昧さ無し。
  - 100%超えは事後。事前特徴で絞れるかは step2(別)。まず本数を見る。
  - 数本しか無い式/帯は「存在するが絞れん」で示唆止まり。

例: python -u -m src.test_fullcover
"""
from __future__ import annotations

import sqlite3
import os

import numpy as np
import pandas as pd

from . import storage

# 賭式: (payoutカラム名, 組数, ラベル)
BETS = [
    ("tansho_yen", 6, "単勝"),
    ("quinella_yen", 15, "2連複"),
    ("exacta_yen", 30, "2連単"),
    ("trio_yen", 20, "3連複"),
    ("trifecta_yen", 120, "3連単"),
]


def main():
    db = os.path.join(storage.DATA_DIR, "kyotei.db")
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT * FROM payouts", con)
    con.close()
    print(f"払戻台帳 {len(df):,}レース\n")

    print(f"{'賭式':<7}{'組数':>5}{'N':>8}{'平均回収':>9}{'中央値':>8}{'控除+全買FLB損':>13}"
          f"{'全買>100%':>10}{'割合':>7}{'最大回収':>10}")
    rows = []
    for col, ncomb, lb in BETS:
        s = df[col].dropna()
        s = s[s > 0]
        if len(s) == 0:
            continue
        ret = s / (ncomb * 100.0)          # 全買い回収率（1点100円×組数）
        over = int((ret > 1.0).sum())
        mean = ret.mean()
        print(f"{lb:<7}{ncomb:>5}{len(s):>8,}{mean*100:>8.1f}%{ret.median()*100:>7.1f}%"
              f"{(1-mean)*100:>12.1f}%{over:>10,}{over/len(s)*100:>6.2f}%{ret.max()*100:>9.0f}%")
        rows.append((lb, ncomb, ret))
    print("\n※平均回収≠払戻率。全買いは全組等額=過剰人気の人気薄に相対過重=FLB最大露出で、")
    print("  実控除(~25%)より深く負ける。『1-平均回収』は控除＋全買いFLB損。")
    print("※全買>100%=勝ち組払戻>組数×100 のレース数＝そのレースだけは総流しで黒字だった（事後）。")

    # 相対効率: 各式で「全買い回収を勝ち組払戻帯で切る」→ どの帯が100%を生むか
    print("\n=== 全買い100%超えは『勝ち組がどのオッズ帯で来た時』か（式別・相対効率）===")
    obands = [(1, 5), (5, 10), (10, 30), (30, 100), (100, 500), (500, 1e9)]
    olbl = ["1-5倍", "5-10", "10-30", "30-100", "100-500", "500倍+"]
    print(f"{'賭式':<7}" + "".join(f"{l:>9}" for l in olbl))
    for lb, ncomb, ret in rows:
        odds = ret * ncomb   # 勝ち組の実効オッズ(=払戻/100)
        cells = []
        for lo, hi in obands:
            m = (odds >= lo) & (odds < hi)
            n = int(m.sum())
            # この帯で全買いが黒字になるのは 勝ち組オッズ>組数 の時
            cells.append(f"{n:>9,}")
        print(f"{lb:<7}" + "".join(cells))
    print("※全買い黒字の条件=勝ち組オッズ>組数(単勝6/2連複15/2連単30/3連複20/3連単120)。"
          "組数を超える帯に的中が集中する式ほど総流し黒字が出やすい。")


if __name__ == "__main__":
    main()
