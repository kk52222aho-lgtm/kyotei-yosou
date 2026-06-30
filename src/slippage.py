"""実弾スリッページ検証（最後の穴＝自分の賭けでプールが動く分を定量化）。

boatrace は締切時オッズ(=発売票数集計完了時点)がそのまま払戻＝表示と実払戻のギャップ無し。
本戦略はオッズでなくモデル信号で引くので、残る実弾リスクは「自分の賭けによるプール変動」だけ。

各勝ち艇の単勝プール内シェアは払戻から逆算: 既存投票 W = 0.75*P/R （P=総プール, R=払戻倍率,
0.75=控除率25%後）。賭け額 S を入れると新払戻 = 0.75*(P+S)/(W+S)。これでクリーン3年の
コントラリアン単勝ベットの回収率を、プール額×賭け額の格子で再計算する。

例:
  python -m src.slippage
"""
from __future__ import annotations

import pandas as pd

from .validate import load, fit_predict_leakfree, SELECTION_YEARS


def collect_bets() -> pd.DataFrame:
    """クリーン年(戦略選択に未使用)の単勝コントラリアンの (won, R=払戻倍率)。"""
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
        for _, g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["tansho_lane"]):
                continue
            won = int(top["tansho_lane"]) == int(top["lane"])
            bets.append((won, top["tansho_yen"] / 100.0))
    return pd.DataFrame(bets, columns=["won", "R"])


def eroded_roi(bets: pd.DataFrame, pool: float, stake: float) -> float:
    """単勝プール pool 円・1レース stake 円賭けたときの回収率。"""
    ret = 0.0
    for won, R in zip(bets.won, bets.R):
        if not won:
            continue
        W = 0.75 * pool / R                       # 勝ち艇への既存投票額
        ret += stake * 0.75 * (pool + stake) / (W + stake)
    return ret / (len(bets) * stake)


def main():
    bets = collect_bets()
    base = bets.loc[bets.won, "R"].sum() / len(bets)
    print(f"クリーン3年 単勝コントラリアン {len(bets)}ベット / 的中{bets.won.mean():.1%}")
    print(f"スリッページ無し回収率: {base*100:.1f}%\n")
    pools = [200000, 500000, 1000000, 3000000]
    stakes = [1000, 3000, 10000, 30000]
    print("=== スリッページ込み回収率（単勝プール × 1レース賭け額）===")
    print(f"{'プール＼賭け':<12}" + "".join(f"{s:>9,}円" for s in stakes))
    for P in pools:
        row = "".join(f"{eroded_roi(bets, P, S)*100:>9.0f}%" for S in stakes)
        print(f"{P:>10,}円 " + row)
    print("\n※少額(1-3千円/レース)なら薄プールでもエッジ生存。大口は自分でオッズを潰す。")


if __name__ == "__main__":
    main()
