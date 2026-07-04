"""対称性の洗い出し：本命(低オッズ)側の割安拾いは、テール(高オッズ)側にも延びとるか。

問い: 全30出目をオッズ帯にバケットして、
  (1) 実現勝率(結果=真値) が market-implied(1/オッズ) を上回る帯はあるか＝FLBの逆か
  (2) model が「割安」と名指す(edge比 p_model/p_market>1)部分集合の実現勝率が、
      その帯の market-implied を上回るか＝割安拾いがテールに延びとる証拠
規律:
  - p_market は締切オッズ由来＝②で潰した29%乖離が高オッズ帯で暴れる。
    →信頼オッズ(勝ち組締切×100≈確定±10%)で切った版と全版の両方を出す。
  - 各帯の勝ち数(事象数)を必ず出す。数個しか勝ってない帯は「示唆」止まり。
  - これは信号の有無(対称性の答え)を見るだけ。賭ける判断は締切/None潰しが先。

例: python -u -m src.test_tail_edge
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .strategy_search import exacta_probs
from .test_value_check import EX_CACHE
from .validate import load, fit_predict_leakfree, SELECTION_YEARS

# market オッズでのバケット（倍率）
OBANDS = [(1, 5), (5, 10), (10, 30), (30, 100), (100, 1e9)]
OLBL = ["1-5倍", "5-10倍", "10-30倍", "30-100倍", "100倍+"]


def collect_combos(df):
    ex_cache = json.load(open(EX_CACHE, encoding="utf-8"))
    pay = {}
    for _, r in df[["date", "jcd", "rno", "exacta_combo", "exacta_yen"]].dropna(subset=["exacta_yen"]).iterrows():
        pay[f"{r['date']}-{r['jcd']}-{int(r['rno'])}"] = (r["exacta_combo"], int(r["exacta_yen"]))

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
                continue  # 妙味レース(本命≠1号艇)のみ＝本命側と同じ母集団
            s = g["p"].sum()
            if s <= 0:
                continue
            key = f"{date}-{jcd}-{rno}"
            ex = ex_cache.get(key)
            if not ex or len(ex) < 30:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            if len(p) != 6:
                continue  # 全6艇スコア済みのみ（impute=Falseで欠けた艇があると30組揃わず正規化が崩れる）
            pm = dict(exacta_probs(p))  # {combo: p_model} 全30
            if len(pm) != 30 or any(ex.get(c, 0) <= 0 for c in pm):
                continue  # 全30組にオッズが揃うレースのみ＝市場を30組で正規化しΣp_market=1に載せる
            allinv = sum(1.0 / ex[c] for c in pm)  # overround（30組ちょうど）
            win = top["exacta_combo"]
            # オッズ信頼度（勝ち組締切×100≈確定±10%）
            trust = False
            if key in pay:
                wc, wy = pay[key]
                wc_odds = ex.get(wc)
                if wc_odds and wy > 0:
                    trust = abs(wc_odds * 100 - wy) / wy <= 0.10
            for combo in pm:
                o = ex[combo]
                p_mkt = (1.0 / o) / allinv          # 控除率フリー market-implied（Σ=1）
                p_mdl = pm[combo]
                rows.append((o, p_mdl, p_mkt, p_mdl / p_mkt if p_mkt > 0 else 0,
                             1 if combo == win else 0, trust))
    return pd.DataFrame(rows, columns=["odds", "p_model", "p_market", "edge", "won", "trust"])


def report(d, label):
    # 保存則の自己検算: 全出目でΣ実現=Σ市場(=レース数)でないと正規化が壊れとる
    n_races = len(d) // 30
    cons = f"[保存則: Σ実現={d.won.sum():,} / Σ市場={d.p_market.sum():.0f} / レース数={n_races:,}]"
    print(f"\n===== {label}（{len(d):,}出目・{d.won.sum():,}勝）{cons} =====")
    print(f"{'オッズ帯':<9}{'出目数':>8}{'勝数':>6}{'実現勝率':>9}{'市場implied':>11}{'model':>8}"
          f"{'実現/市場':>9}   ｜ model割安(edge>1)のみ: {'出目':>6}{'勝':>5}{'実現':>7}{'市場':>7}")
    for (lo, hi), lb in zip(OBANDS, OLBL):
        sub = d[(d.odds >= lo) & (d.odds < hi)]
        if len(sub) == 0:
            continue
        real = sub.won.mean()
        mkt = sub.p_market.mean()
        mdl = sub.p_model.mean()
        up = sub[sub.edge > 1]      # model が割安と名指した部分集合
        up_real = up.won.mean() if len(up) else float("nan")
        up_mkt = up.p_market.mean() if len(up) else float("nan")
        flag = ""
        if int(sub.won.sum()) < 20:
            flag = " ※勝<20=示唆"
        print(f"{lb:<9}{len(sub):>8,}{int(sub.won.sum()):>6}{real*100:>8.2f}%{mkt*100:>10.2f}%"
              f"{mdl*100:>7.2f}%{real/mkt if mkt else 0:>8.2f}   ｜{len(up):>13,}{int(up.won.sum()):>5}"
              f"{up_real*100:>6.2f}%{up_mkt*100:>6.2f}%{flag}")


def main():
    df = load()
    d = collect_combos(df)
    print(f"収集: 妙味レースの全30出目 {len(d):,}行")
    print("読み方: 実現/市場>1=その帯は市場より当たる(FLBの逆)。"
          "model割安(edge>1)の実現>市場=割安拾いがテールに延びとる証拠。勝<20帯は示唆止まり。")
    report(d, "全レース（締切オッズそのまま＝29%乖離込み）")
    report(d[d.trust], "信頼オッズのみ（勝ち組締切×100≈確定±10%）")
    print("\n※market-impliedは締切オッズ由来。高オッズ帯ほどオッズ精度に敏感＝賭ける判断は締切/None潰しが先。")
    print("※テール事象はスパース＝edge比が1超えても『信号の示唆』であって『抽出可能』の証明ではない。")


if __name__ == "__main__":
    main()
