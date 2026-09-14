"""市場ミスプライス(EV)の総ざらい: 予測は天井、残る勝ち筋はEV割安のみ＝全部机上検証。

手持ちの締切オッズ(exacta_odds_cache=2連単, odds_cache=単勝)で、妙味(本命≠1号)クリーン年に:
  単勝本命EV / 2連単top3EV を細かく閾値スイープ → ROI・的中・年別・頑健性(上位除外)。
どのEV割安ラインthaが一番効くか、年またぎで生きるか、fat-tailか、を一望する。
3連系は履歴オッズ無し(=live収集で今後)。確定払戻・walk-forward・リークなし(本命EVは本命の締切odズ=事前値)。

例: python -u -m src.test_ev_full
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


def collect(df):
    tan = json.load(open(TAN, encoding="utf-8")) if os.path.exists(TAN) else {}
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
            if int(top["lane"]) == 1 or pd.isna(top["tansho_yen"]):
                continue
            key = f"{date}-{jcd}-{rno}"
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            honmei = int(top["lane"])
            won_t = int(top["tansho_lane"]) == honmei
            to = tan.get(key)
            h_odds = to.get(str(honmei)) if to else None
            tan_ev = p[honmei] * h_odds if h_odds else None
            tan_ret = float(top["tansho_yen"]) if won_t else 0.0
            ex = exc.get(key)
            ex_ev = ex_ret = None
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            pmap = dict(exacta_probs(p))
            if ex and all(ex.get(c) for c in top3):
                ex_ev = sum(pmap[c] * ex[c] for c in top3) / 3.0
                ex_win = top["exacta_combo"] in top3
                ex_ret = float(top["exacta_yen"]) if (ex_win and not pd.isna(top["exacta_yen"])) else 0.0
            rows.append({"yr": y, "tan_ev": tan_ev, "tan_ret": tan_ret,
                         "ex_ev": ex_ev, "ex_ret": ex_ret})
    return rows


def sweep(rows, evkey, retkey, stake_per, ths, label):
    print(f"\n=== {label}：EV割安ライン・スイープ（1点{stake_per}円）===")
    print(f"  {'EV>':>5}{'N':>6}{'ROI':>8}{'的中':>7}  年別ROI")
    for th in ths:
        sub = [r for r in rows if r.get(evkey) is not None and r[evkey] > th and r.get(retkey) is not None]
        if not sub:
            continue
        n = len(sub)
        ret = sum(r[retkey] for r in sub)
        roi = ret / (n * stake_per)
        yb = {}
        for r in sub:
            yb.setdefault(r["yr"], [0.0, 0])
            yb[r["yr"]][0] += r[retkey]
            yb[r["yr"]][1] += stake_per
        ys = " ".join(f"{y}:{v[0]/v[1]*100:.0f}%" for y, v in sorted(yb.items()))
        hit = sum(1 for r in sub if r[retkey] > 0) / n
        print(f"  {th:>5.1f}{n:>6}{roi*100:>7.0f}%{hit*100:>6.0f}%  {ys}")


def main():
    df = load()
    rows = collect(df)
    print(f"妙味(本命≠1号)クリーン年 {len(rows):,}レース｜"
          f"単勝EVあり{sum(1 for r in rows if r['tan_ev'] is not None):,}｜"
          f"2連単EVあり{sum(1 for r in rows if r['ex_ev'] is not None):,}")
    sweep(rows, "tan_ev", "tan_ret", 100, [1.0, 1.1, 1.2, 1.3, 1.5, 2.0], "単勝(本命)")
    sweep(rows, "ex_ev", "ex_ret", 300, [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0], "2連単(上位3点)")
    hb = [r for r in rows if r.get("ex_ev") is not None and r["ex_ev"] > 2.0 and r.get("ex_ret") is not None]
    if hb:
        rets = np.sort(np.array([r["ex_ret"] for r in hb]))[::-1]
        stake = len(hb) * 300
        print(f"\n  【2連単EV>2.0 fat-tail】{len(hb)}件 上位除外ROI: "
              + " ".join(f"{k}本->{rets[k:].sum()/stake*100:.0f}%" for k in [0, 1, 5, 20, 40]))
    print("\n※EVが上がるほどROIも上がる&年別で生存 なら市場ミスプライスは本物の割安拾い。")
    print("  3連系は履歴オッズ無し＝live収集(着火済)で今後。確定払戻・ライブ未証明(締切29%乖離は割引く)。")


if __name__ == "__main__":
    main()
