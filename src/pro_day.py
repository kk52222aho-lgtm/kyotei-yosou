"""プロ舟券師の一日: サンプル日の全レースを『naive全買い vs プロ選択』で計算しJSON化。

狙い: 「一日全レース買ったら負ける／プロは妙味だけ選ぶ」を実データで見せる。
  walk-forward(train<2025)でリークなし。payoutは確定(DB)。data/pro_day.json に保存。

例: python -u -m src.pro_day
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import storage
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree

OUT = os.path.join(storage.DATA_DIR, "pro_day.json")
from .venues import name as venue_name


def main():
    df = load()
    # 2025で「exacta払戻あり・レース数が多い」日を1つ選ぶ
    d = df[(df["yr"] == "2025") & df["exacta_yen"].notna()]
    cnt = d.groupby("date")["rno"].nunique()  # 実質は (date) 単位のレース数近似
    date = d.groupby("date").size().sort_values(ascending=False).index
    # レース数(=行/6 の多い)日
    best = (d.groupby("date").size() // 6).sort_values(ascending=False)
    DATE = best.index[0]

    te = (df["date"] == DATE).to_numpy()
    tr = (df["yr"].astype(int) < 2025).to_numpy()
    test = fit_predict_leakfree(df, tr, te)

    races = []
    for (dt_, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        top = g.iloc[0]
        if pd.isna(top["exacta_yen"]) or pd.isna(top["tansho_lane"]):
            continue
        honmei = int(top["lane"])
        s = g["p"].sum()
        if s <= 0:
            continue
        p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
        top3 = [c for c, _ in exacta_probs(p)[:3]]
        won_t = int(top["tansho_lane"]) == honmei
        t_ret = int(top["tansho_yen"]) if won_t else 0
        e_win = top["exacta_combo"] in top3
        e_ret = int(top["exacta_yen"]) if e_win else 0
        races.append({
            "venue": venue_name(jcd), "rno": int(rno),
            "honmei": honmei, "contrarian": honmei != 1,
            "winner": int(top["tansho_lane"]),
            "won_t": won_t, "t_ret": t_ret,
            "top3": top3, "e_combo": top["exacta_combo"], "e_win": e_win, "e_ret": e_ret,
            "pl": (t_ret - 100) + (e_ret - 300),
        })

    def totals(rs):
        n = len(rs)
        t_st = n * 100; t_ret = sum(r["t_ret"] for r in rs)
        e_st = n * 300; e_ret = sum(r["e_ret"] for r in rs)
        return {"races": n, "t_ret": t_ret, "e_ret": e_ret,
                "stake": t_st + e_st, "ret": t_ret + e_ret,
                "pl": (t_ret + e_ret) - (t_st + e_st),
                "roi": (t_ret + e_ret) / (t_st + e_st) if (t_st + e_st) else 0,
                "t_hit": sum(1 for r in rs if r["won_t"]),
                "e_hit": sum(1 for r in rs if r["e_win"])}

    pro = [r for r in races if r["contrarian"]]
    out = {
        "date": DATE,
        "naive": totals(races),          # 全レース買い（本命単勝+2連単3点）
        "pro": totals(pro),              # 妙味(本命≠1号艇)だけ
        "races": races,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{DATE}: 全{len(races)}レース / 妙味(本命≠1号艇){len(pro)}レース")
    print(f"  naive全買い: 回収{out['naive']['roi']*100:.0f}% 収支{out['naive']['pl']:+,}円")
    print(f"  プロ選択   : 回収{out['pro']['roi']*100:.0f}% 収支{out['pro']['pl']:+,}円")
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
