"""逐次ベッティングエージェント（結果を見る前に1レースずつ賭ける・複利・月別正直集計）。

狙い: 目標を持って攻めるが、月別P&Lは負け月込みで正直に出す。「毎月30万」が
  現実に何回起きて最悪どこまで沈むかを数字で突きつける。walk-forward=先読みなし。

戦略（三関門＋資金%張り＝複利で攻める）:
  1. 妙味(本命≠1号艇) かつ 単勝≥1.5 → 本命単勝に資金の TAN_PCT
  2. さらに 2連単EV>2.0 → 上位3点に資金の EX_PCT（EVが高いほど上げる=攻め）
  資金は各ベットの%で張るので勝てば張り額も増える（複利）。
正直な釘: 締切オッズは29%乖離あり=EVは信頼寄りで割引く。エッジはライブ未証明=これは机上の分布。

例: python -u -m src.agent
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import storage
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS

TAN = os.path.join(storage.DATA_DIR, "odds_cache.json")
EX = os.path.join(storage.DATA_DIR, "exacta_odds_cache.json")

START = 300_000       # 初期資金30万（=ユーザーの月目標を元手に）
TAN_PCT = 0.01        # 単勝: 資金の1%
EX_BASE = 0.02        # 2連単: 基本 資金の2%
EX_KELLY = 0.02       # +EV連動（EV-2）×これ、で攻め増し
EX_CAP = 0.06         # 2連単1レース上限 資金の6%
EV_TH = 2.0


def collect(df):
    tan = json.load(open(TAN, encoding="utf-8")) if os.path.exists(TAN) else {}
    ex_c = json.load(open(EX, encoding="utf-8")) if os.path.exists(EX) else {}
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
            if int(top["lane"]) == 1 or pd.isna(top["tansho_yen"]):
                continue
            key = f"{date}-{jcd}-{rno}"
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            honmei = int(top["lane"])
            won_t = int(top["tansho_lane"]) == honmei
            if won_t:
                tan_odds = top["tansho_yen"] / 100.0
            else:
                od = tan.get(key)
                tan_odds = od.get(str(honmei)) if od else None
            if tan_odds is None:
                continue
            # 2連単 EV
            ex = ex_c.get(key)
            ev = ex_win = ex_odds_win = None
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            if ex and all(ex.get(c) for c in top3):
                pmap = dict(exacta_probs(p))
                ev = sum(pmap[c] * ex[c] for c in top3) / 3.0
                ex_win = top["exacta_combo"] in top3
                ex_odds_win = (int(top["exacta_yen"]) / 100.0) if ex_win else 0
            rows.append({
                "date": date, "tan_odds": float(tan_odds), "won_t": won_t,
                "tan_ret_per100": (top["tansho_yen"] / 100.0) if won_t else 0.0,
                "ev": ev, "ex_win": bool(ex_win) if ex_win is not None else None,
                "ex_ret_per100": ex_odds_win if ex_win else 0.0,
            })
    rows.sort(key=lambda r: r["date"])   # 時系列順（逐次）
    return rows


def main():
    df = load()
    rows = collect(df)
    # 定額張り（複利の幻を消す）: 1点あたり UNIT 円。単勝1点＋2連単3点(EV>2.0時)
    UNIT = 1000
    SLIP = 0.90        # スリッページ概算: 大きめ張ると薄プールで払戻が目減り(検証で大口81%)。控えめに-10%
    print(f"妙味レース {len(rows)} を時系列順に逐次ベット（定額 {UNIT:,}円/点・スリッページ{int((1-SLIP)*100)}%控除）\n")

    monthly = {}
    n_bet = 0
    cum = 0
    for r in rows:
        if r["tan_odds"] < 1.5:
            continue
        mo = r["date"][:6]
        m = monthly.setdefault(mo, [0.0, 0])
        pl = 0.0
        # 単勝1点
        pl -= UNIT
        if r["won_t"]:
            pl += UNIT * r["tan_ret_per100"] * SLIP
        # 2連単3点（EV>2.0）
        if r["ev"] is not None and r["ev"] > EV_TH:
            pl -= UNIT * 3
            if r["ex_win"]:
                pl += UNIT * r["ex_ret_per100"] * SLIP
        m[0] += pl
        m[1] += 1
        cum += pl
        n_bet += 1

    a = np.array([v[0] for _, v in sorted(monthly.items())])
    print(f"{'月':<9}{'収支':>12}{'ベット':>7}{'累計':>14}")
    run = 0
    for mo in sorted(monthly):
        pl, nb = monthly[mo]
        run += pl
        mark = " ✅30万+" if pl >= 300_000 else (" 🔴赤字" if pl < 0 else "")
        print(f"{mo[:4]}/{mo[4:]:<4}{int(pl):>+12,}{nb:>7}{int(run):>+14,}{mark}")

    win = int((a > 0).sum()); hit30 = int((a >= 300_000).sum())
    print(f"\n=== 正直な現実（定額{UNIT:,}円/点・3年{n_bet}ベット）===")
    print(f"  通算 {int(cum):+,}円 / 月平均 {int(a.mean()):+,}円")
    print(f"  勝ち月 {win}/{len(a)}（{win/len(a)*100:.0f}%）・負け月 {len(a)-win} / 最悪月 {int(a.min()):+,}円")
    print(f"  『毎月30万+』達成 {hit30}/{len(a)}ヶ月（{hit30/len(a)*100:.0f}%）")
    print("\n※定額張りにして複利の幻を消した。スリッページ-10%は控えめ＝実際はもっと食われる可能性。")
    print("※これでも机上（過去データ・締切オッズ29%乖離込み・ライブ未証明）。来月出る保証はゼロ。")
    print("※これが『必死に攻めても、正直に出すとこうなる』の姿。負け月も最悪月も隠さん。")


if __name__ == "__main__":
    main()
