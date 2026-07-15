"""歪みの再探索: Favorite-Longshot Bias(本命-大穴バイアス)を全艇×確率帯で測る。

今まで「モデル本命を買う」ROIしか見てなかった=歪みの片側しか見てない。パリミュチュエル
世界共通の歪み=客は大穴を買いすぎ(配当が実力以下)・本命を買わなさすぎ(配当が実力以上)。
∴全6艇をモデル確率の帯に割り、各帯の単勝ROI・実勝率・較正を測る。控除25%を超える帯=
市場が過小評価してる本物の歪み。特に超本命帯(model_p高)が>100%なら本命バイアスの妙味。
確定払戻・walk-forward・6艇クリーン。全艇賭けの各帯ROI=勝者配当/(頭数×100)。

例: python -u -m src.test_flb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df):
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            win_lane = int(g.iloc[0]["tansho_lane"])
            yen = float(g.iloc[0]["tansho_yen"])
            for r in g.itertuples():
                won = int(r.lane) == win_lane
                rows.append({
                    "mp": float(r.p) / s,                    # レース内正規化 model確率
                    "won": int(won),
                    "ret": yen if won else 0.0,              # 勝者のみ確定単勝配当
                })
    return pd.DataFrame(rows)


def main():
    df = load()
    d = collect(df)
    print(f"全艇 総数 {len(d):,}（6艇クリーン・確定払戻）\n")

    # モデル確率の帯(FLB曲線)。各帯: 頭数/実勝率/平均model_p(較正)/ROI/含意オッズ
    edges = [0, .02, .05, .08, .12, .18, .25, .35, .45, .55, .70, 1.01]
    d["bkt"] = pd.cut(d["mp"], edges, right=False)
    print("model確率帯 :   頭数   実勝率  平均p(較正)  ROI(単勝)   ※ROI>100=歪み")
    for b, sub in d.groupby("bkt", observed=True):
        n = len(sub)
        if n < 200:
            continue
        wr = sub["won"].mean() * 100
        mp = sub["mp"].mean() * 100
        roi = sub["ret"].sum() / (n * 100) * 100
        flag = "  ← 控除超え！" if roi >= 100 else ("  ← 惜" if roi >= 95 else "")
        print(f"  [{b.left:.2f},{b.right:.2f}): {n:7d}  {wr:5.1f}%  {mp:5.1f}%     {roi:5.1f}%{flag}")

    # 超本命だけ深掘り(本命バイアスは最上端に出やすい)
    print("\n[超本命帯 深掘り: model_p 上位]")
    for lo in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = d[d["mp"] >= lo]
        if len(sub) < 100:
            continue
        roi = sub["ret"].sum() / (len(sub) * 100) * 100
        print(f"  model_p≥{lo:.2f}: 頭数{len(sub):6d}  実勝率{sub['won'].mean()*100:5.1f}%  ROI{roi:5.1f}%")

    print("\n読み: どの帯もROI~75-80%(=1-控除)で平ら → 純粋に控除だけ=歪み無し・効率的。")
    print("      本命帯that大穴帯より明確に高い/どこかで>100% → Favorite-Longshot Bias=本物の歪み。")


if __name__ == "__main__":
    main()
