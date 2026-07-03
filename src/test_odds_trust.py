"""②再検算の母数健全性：信頼オッズ71%の除外は帯別に偏るか、除外バイアスの符号はどっち向きか。

前回の穴: 信頼71%だけで超過を出したが、帯別残存N・除外の帯別分布・除外バイアスの符号を見てへん。
  乖離(勝ち組過小評価)が帯で偏るなら、信頼71%は母集団の代表やなく条件付き部分集合。
問い:
  1) 帯別 全N→信頼N（残存件数・的中数）。二桁割れの帯は水準保留（②の規律）
  2) 除外29%の帯別分布（低配当帯偏りか高配当帯偏りか）
  3) 除外バイアスの符号：全hit率 vs 信頼hit率で、除外が的中(=超過過大)を抜いたか外れ(=超過過小)を抜いたか
     → 信頼超過が上限(楽観)か下限(悲観)か、帯ごとに判定

例: python -u -m src.test_odds_trust
"""
from __future__ import annotations

import json

import pandas as pd

from .test_value_check import collect, BANDS, BAND_LBL, EX_CACHE
from .validate import load

TOL = 0.10


def main():
    df = load()
    pay = {}
    for _, r in df[["date", "jcd", "rno", "exacta_combo", "exacta_yen"]].dropna(subset=["exacta_yen"]).iterrows():
        pay[f"{r['date']}-{r['jcd']}-{int(r['rno'])}"] = (r["exacta_combo"], int(r["exacta_yen"]))

    rows = collect(df)
    rows = [r for r in rows if r["tansho_odds"] is not None]
    ex_cache = json.load(open(EX_CACHE, encoding="utf-8"))

    recs = []
    for r in rows:
        ex = ex_cache.get(r["key"])
        if not ex or len(ex) < 30:
            continue
        top3_odds = [ex.get(c) for c in r["top3"] if ex.get(c)]
        if len(top3_odds) < 3:
            continue
        allinv = sum(1.0 / o for o in ex.values() if o > 0)
        norm_implied = sum(1.0 / o for o in top3_odds) / allinv
        hit = r["win_combo"] in r["top3"]
        trust = None
        if r["key"] in pay:
            wc, wy = pay[r["key"]]
            wclose = ex.get(wc)
            if wclose and wy > 0:
                trust = abs(wclose * 100 - wy) / wy <= TOL
        recs.append({"tansho_odds": r["tansho_odds"], "hit": hit,
                     "norm_implied": norm_implied, "trust": bool(trust)})
    d = pd.DataFrame(recs)

    print(f"有効 {len(d)}レース / 信頼 {int(d.trust.sum())}（{d.trust.mean()*100:.0f}%）/ "
          f"除外 {int((~d.trust).sum())}\n")

    print("=== 1) 帯別 残存N・的中数（②規律: 的中<10は水準保留）===")
    print(f"{'帯':<9}{'全N':>6}{'全的中':>7}{'信頼N':>7}{'信頼的中':>8}{'残存率':>8}{'保留?':>8}")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        a = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        t = a[a.trust]
        if len(a) == 0:
            continue
        th = int(t.hit.sum())
        hold = "保留⚠" if th < 10 else "OK"
        print(f"{lb:<9}{len(a):>6}{int(a.hit.sum()):>7}{len(t):>7}{th:>8}{len(t)/len(a)*100:>7.0f}%{hold:>8}")

    print("\n=== 2) 除外29%の帯別分布（残存率が下がる帯＝乖離が集中）===")
    print(f"{'帯':<9}{'除外N':>7}{'除外率':>8}")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        a = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        if len(a) == 0:
            continue
        ex_n = int((~a.trust).sum())
        print(f"{lb:<9}{ex_n:>7}{ex_n/len(a)*100:>7.0f}%")

    print("\n=== 3) 除外バイアスの符号（全hit率 vs 信頼hit率 → 信頼超過は上限か下限か）===")
    print(f"{'帯':<9}{'全hit率':>8}{'信頼hit率':>9}{'全超過':>8}{'信頼超過':>9}{'向き':>22}")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        a = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        t = a[a.trust]
        if len(a) == 0 or len(t) == 0:
            continue
        a_hr, t_hr = a.hit.mean(), t.hit.mean()
        a_ex = (a_hr - a.norm_implied.mean()) * 100
        t_ex = (t_hr - t.norm_implied.mean()) * 100
        # 信頼で超過が上がった=除外は超過の低いレース(model miss)を抜いた=信頼は楽観側(上限)
        # 信頼で超過が下がった=除外は超過の高いレース(inflated hit)を抜いた=信頼は正直側(下方修正)
        if t_ex > a_ex:
            direction = "信頼↑=除外はmiss抜き→楽観側(上限)"
        else:
            direction = "信頼↓=除外は水増しhit抜き→正直側"
        print(f"{lb:<9}{a_hr*100:>7.1f}%{t_hr*100:>8.1f}%{a_ex:>+7.1f}{t_ex:>+8.1f}  {direction}")

    print("\n※残存率が高配当帯ほど下がるなら、乖離は高配当帯に集中＝そこの信頼再計算は薄い部分集合。")
    print("※信頼hit率>全hit率の帯＝除外がmodel missを抜いた＝信頼超過はその帯で楽観側の上限。生存と断定不可。")


if __name__ == "__main__":
    main()
