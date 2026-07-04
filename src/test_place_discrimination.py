"""3連単の土台チェック：モデルは1着だけでなく2着・3着を見分けられるか(オッズ不要)。

問い: bet-eligible subset(本命≠1号艇)で、モデルの順位付けが実際の2着/3着方向にも
  信号を持つか。marginalな艇スコアの掛け算(Plackett-Luce)が意味を持つ前提=下位順位の弁別。
  これがnaive(nat_win/lane)を超えなければ、3連単basketは順列確率の土台が無い=着手前に負け確。
規律:
  - walk-forward OOF。bet subset(本命≠1号艇)に限定=実際に張る母集団。
  - naive基準(nat_win順・lane順)と必ず比較。モデル単独の数字は無意味。
  - 「1着は当たるが下位は当たらん」なら3連単は土台なし。下位でもnaive超えなら土台あり。
  - オッズ一切不要=fat-tail ROI/控除率の罠に触れず、生の弁別だけ測る。

例: python -u -m src.test_place_discrimination
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .validate import load, fit_predict_leakfree, SELECTION_YEARS


def order_by(g, col, asc):
    return list(g.sort_values(col, ascending=asc)["lane"])


def main():
    df = load()
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        test["fin"] = pd.to_numeric(test["finish"], errors="coerce")
        for _, g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            if int(g.iloc[0]["lane"]) == 1:
                continue  # bet subset: 本命≠1号艇
            fin = g["fin"].to_numpy()
            if len(g) != 6 or not np.array_equal(np.sort(fin), np.arange(1, 7)):
                continue  # 6艇・着順1-6が揃うレースのみ
            rows.append(g)
    print(f"bet subset(本命≠1号艇・6艇完走) {len(rows):,}レース\n")

    # 各方式の「予測k位の艇が実際にk着か」＋Spearman＋条件付き下位弁別
    def eval_order(rows, col, asc, name):
        pos_hit = {1: 0, 2: 0, 3: 0}
        sp = []
        c2 = c2d = c3 = c3d = 0  # 条件付き2着/3着弁別
        top3set = 0
        for g in rows:
            order = order_by(g, col, asc)          # 予測順(1位→6位)
            fin = dict(zip(g["lane"], g["fin"]))
            actual = sorted(fin, key=lambda l: fin[l])  # 実際の着順(1着→)
            for k in (1, 2, 3):
                if fin[order[k - 1]] == k:
                    pos_hit[k] += 1
            # Spearman(予測順位 vs 実着順)
            pred_rank = {l: i + 1 for i, l in enumerate(order)}
            r, _ = spearmanr([pred_rank[l] for l in g["lane"]], list(g["fin"]))
            if not np.isnan(r):
                sp.append(r)
            # 条件付き2着: 実1着を除いた中でpred最上位=実2着か
            w1 = actual[0]
            rest = [l for l in order if l != w1]
            if rest:
                c2d += 1
                if rest[0] == actual[1]:
                    c2 += 1
            # 条件付き3着: 実1・2着を除いた中でpred最上位=実3着か
            rest2 = [l for l in order if l not in (actual[0], actual[1])]
            if rest2:
                c3d += 1
                if rest2[0] == actual[2]:
                    c3 += 1
            # top3集合一致(3連複box)
            if set(order[:3]) == set(actual[:3]):
                top3set += 1
        n = len(rows)
        print(f"{name:<10}"
              f"{pos_hit[1]/n*100:>7.1f}%{pos_hit[2]/n*100:>7.1f}%{pos_hit[3]/n*100:>7.1f}%"
              f"{np.mean(sp):>9.3f}"
              f"{c2/c2d*100:>9.1f}%{c3/c3d*100:>9.1f}%{top3set/n*100:>9.1f}%")

    print(f"{'方式':<10}{'1着的中':>7}{'2着的中':>7}{'3着的中':>7}{'Spearman':>9}"
          f"{'条件付2着':>9}{'条件付3着':>9}{'top3集合':>9}")
    eval_order(rows, "p", False, "モデル")
    eval_order(rows, "nat_win", False, "nat_win")
    eval_order(rows, "lane", True, "lane順")
    print("\n★決定打=無条件marginalの2着/3着的中(全レース同一母集団)。ここでnat_winに上乗せゼロ=土台なし。")
    print("※条件付2着/3着は【解釈注意】: モデルとnat_winで条件通過レース群(母集団)が違う=難易度差を")
    print("  順位能力差と誤読する。決定打に使うな。無条件marginalで判定せよ。")
    print("※機構: 1着二値学習の勾配は1着/非1着境界にしか効かず、モデルは2着3着を見てすらいない。")
    print("※nat_win≈市場implied(crowdも同じ強さ順で値付け)。順序モデルがnat_winしか届かねば市場なぞり=EV無。")


if __name__ == "__main__":
    main()
