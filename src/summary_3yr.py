"""現行戦略のフル3年（クリーン年2023-25）成績サマリ。

オッズ不要でデータから正確に出せる部分:
  - 妙味レース(本命≠1号艇)の総数
  - 単勝(本命1点) の的中率・回収率
  - 2連単(確率上位3点) の的中率・回収率
  - 合計

単勝<1.5の丸ごと見送りフィルタはオッズが要る（負けレースのオッズ未取得）ため、
ここでは「フィルタ無し」の全数集計。フィルタ効果は別途サンプル検証(test_*)参照。

例:
  python -m src.summary_3yr
"""
from __future__ import annotations

import pandas as pd

from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS


def year_stats(test: pd.DataFrame) -> dict:
    n = t_hit = t_ret = 0
    e_races = e_hit = e_ret = 0
    for _, g in test.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        top = g.iloc[0]
        if int(top["lane"]) == 1:
            continue  # 妙味レースのみ(本命≠1号艇)
        # 単勝
        if not pd.isna(top["tansho_lane"]):
            n += 1
            if int(top["tansho_lane"]) == int(top["lane"]):
                t_hit += 1
                t_ret += top["tansho_yen"]
        # 2連単 上位3点
        if not pd.isna(top["exacta_yen"]):
            s = g["p"].sum()
            if s > 0:
                p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
                top3 = [c for c, _ in exacta_probs(p)[:3]]
                e_races += 1
                if top["exacta_combo"] in top3:
                    e_hit += 1
                    e_ret += top["exacta_yen"]
    return {"races": n, "t_hit": t_hit, "t_ret": t_ret,
            "e_races": e_races, "e_hit": e_hit, "e_ret": e_ret}


def main():
    df = load()
    years = [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]

    agg = {"races": 0, "t_hit": 0, "t_ret": 0, "e_races": 0, "e_hit": 0, "e_ret": 0}
    per = {}
    for y in years:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        s = year_stats(test)
        per[y] = s
        for k in agg:
            agg[k] += s[k]

    def roi_t(s):
        return s["t_ret"] / (s["races"] * 100) if s["races"] else 0

    def roi_e(s):
        return s["e_ret"] / (s["e_races"] * 3 * 100) if s["e_races"] else 0

    def roi_all(s):
        stake = s["races"] * 100 + s["e_races"] * 3 * 100
        return (s["t_ret"] + s["e_ret"]) / stake if stake else 0

    print("=== 現行戦略・クリーン3年(2023-25) フル成績 / フィルタ無し(全妙味レース) ===\n")
    print(f"{'年':<8}{'妙味R':>7}{'単勝的中':>9}{'単勝回収':>9}{'2連単的中':>10}{'2連単回収':>10}{'合計回収':>9}")
    for y in years:
        if y not in per:
            continue
        s = per[y]
        print(f"{y:<8}{s['races']:>7}"
              f"{s['t_hit']/s['races']*100:>8.1f}%{roi_t(s)*100:>8.1f}%"
              f"{s['e_hit']/s['e_races']*100:>9.1f}%{roi_e(s)*100:>9.1f}%"
              f"{roi_all(s)*100:>8.1f}%")
    s = agg
    print("-" * 62)
    print(f"{'3年計':<8}{s['races']:>7}"
          f"{s['t_hit']/s['races']*100:>8.1f}%{roi_t(s)*100:>8.1f}%"
          f"{s['e_hit']/s['e_races']*100:>9.1f}%{roi_e(s)*100:>9.1f}%"
          f"{roi_all(s)*100:>8.1f}%")

    stake = s["races"] * 100 + s["e_races"] * 3 * 100
    ret = s["t_ret"] + s["e_ret"]
    print(f"\n勝負レース {s['races']} / 賭金合計 {stake:,}円 / 払戻 {ret:,}円 / 収支 {ret-stake:+,}円")
    print("※フィルタ無し(全妙味レース)。単勝<1.5の丸ごと見送りを足すと更に改善(サンプル検証: <1.5群は死に金)。")


if __name__ == "__main__":
    main()
