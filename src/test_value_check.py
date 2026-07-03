"""②割安拾いの検算：モデル2連単的中率 vs 市場implied（クリーン年・締切オッズ）。

問い: 本命単勝オッズ帯ごとに、モデル上位3組の実的中率が市場implied確率を超えとるか。
  超える → 割安拾い(実エッジ)。超えない/控除率で符号が動く → 機械効果(分散チェイス)。

循環回避: 帯は本命単勝オッズで切るが、判定はROIやなくimplied超過。
  ROIは配当込みで循環したが、implied超過は配当を市場評価(締切オッズ)で割り戻すので循環せん。
控除率頑健性: implied=(1-控除率)/オッズ の控除率を 22/25/28% で振り、超過の符号が不変か見る。
控除率フリー従突合: モデル上位3組の的中率 vs 市場最人気3組(最低オッズ3組)の的中率。
  モデルの中人気拾いが市場の鉄板より当たれば、控除率仮定なしで割安拾いの証拠。
規律: 帯の的中数が二桁未満ならその帯は判定保留。

例: python -u -m src.test_value_check
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import scraper, storage
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS

TAN_CACHE = os.path.join(storage.DATA_DIR, "odds_cache.json")        # 単勝(本命帯用)
EX_CACHE = os.path.join(storage.DATA_DIR, "exacta_odds_cache.json")  # 2連単30組
BANDS = [(1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 8.0), (8.0, 1e9)]
BAND_LBL = ["1.0-1.5", "1.5-2.5", "2.5-4", "4-8", "8+"]
TAKEOUTS = [0.22, 0.25, 0.28]


def collect(df):
    tan = json.load(open(TAN_CACHE, encoding="utf-8")) if os.path.exists(TAN_CACHE) else {}
    rows = []
    for y in ["2023", "2024", "2025"]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["exacta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            key = f"{date}-{jcd}-{rno}"
            # 本命単勝オッズ: 勝ち→払戻/100、負け→単勝キャッシュ
            if int(top["tansho_lane"]) == int(top["lane"]):
                t_odds = top["tansho_yen"] / 100.0
            else:
                od = tan.get(key)
                t_odds = od.get(str(int(top["lane"]))) if od else None
            rows.append({
                "key": key, "date": date, "jcd": jcd, "rno": int(rno),
                "tansho_odds": t_odds, "top3": top3,
                "win_combo": top["exacta_combo"],
            })
    return rows


def main():
    df = load()
    rows = collect(df)
    rows = [r for r in rows if r["tansho_odds"] is not None]
    print(f"クリーン年 妙味レース {len(rows)}（本命単勝オッズ取得済）")

    ex_cache = json.load(open(EX_CACHE, encoding="utf-8")) if os.path.exists(EX_CACHE) else {}
    print(f"2連単締切オッズ取得（キャッシュ{len(ex_cache)}）…")
    n_scr = 0
    for i, r in enumerate(rows):
        if r["key"] not in ex_cache:
            ex = scraper.fetch_exacta_odds(r["date"], r["jcd"], r["rno"])
            ex_cache[r["key"]] = ex if ex else {}
            n_scr += 1
            time.sleep(0.3)
            if n_scr % 100 == 0:
                json.dump(ex_cache, open(EX_CACHE, "w", encoding="utf-8"))
                print(f"  scraped {n_scr}")
    json.dump(ex_cache, open(EX_CACHE, "w", encoding="utf-8"))

    # 各レースの指標を作る
    recs = []
    for r in rows:
        ex = ex_cache.get(r["key"])
        if not ex or len(ex) < 30:
            continue
        top3_odds = [ex.get(c) for c in r["top3"] if ex.get(c)]
        if len(top3_odds) < 3:
            continue
        hit = r["win_combo"] in r["top3"]
        allinv = sum(1.0 / o for o in ex.values() if o > 0)   # オーバーラウンド(≒1.33)
        # 主: 控除率フリー。オーバーラウンドで正規化した市場prob(上位3組)
        norm_implied = sum(1.0 / o for o in top3_odds) / allinv
        # 従: 控除率仮定版(感度チェック)
        implied_t = {t: sum((1 - t) / o for o in top3_odds) for t in TAKEOUTS}
        # 市場最人気3組(最低オッズ3組) と 重複率
        fav3 = [c for c, _ in sorted(ex.items(), key=lambda kv: kv[1])[:3]]
        recs.append({
            "tansho_odds": r["tansho_odds"], "hit": hit,
            "norm_implied": norm_implied, **{f"imp_{int(t*100)}": implied_t[t] for t in TAKEOUTS},
            "market_fav3_hit": r["win_combo"] in fav3,
            "overlap": len(set(r["top3"]) & set(fav3)),
            "n_combos": sum(1 for o in ex.values() if o > 0),   # 正規化分母の組数
        })
    d = pd.DataFrame(recs)
    print(f"有効 {len(d)}レース\n")

    print("=== 母数健全性チェック（読む前に）===")
    print(f"{'本命単勝帯':<9}{'レース':>6}{'的中分母':>8}{'市場prob分母':>11}{'平均分母組数':>11}")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        sub = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        if len(sub) == 0:
            continue
        nh = int(sub["hit"].notna().sum())        # モデル的中率の分母
        ni = int(sub["norm_implied"].notna().sum())  # 市場probの分母
        flag = "一致" if nh == ni else "ズレ⚠"
        print(f"{lb:<9}{len(sub):>6}{nh:>8}{ni:>11}{sub.n_combos.mean():>10.1f}/30  {flag}")
    print()

    print("=== ② 主: モデル上位3組 的中率 vs 市場prob(オーバーラウンド正規化・控除率フリー) ===")
    print(f"{'本命単勝帯':<9}{'レース':>6}{'的中':>5}{'モデル的中':>10}{'市場prob':>9}{'超過':>8}")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        sub = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        if len(sub) == 0:
            continue
        hits = int(sub.hit.sum())
        mhr, mkt = sub.hit.mean(), sub.norm_implied.mean()
        hold = " ※保留(的中<10)" if hits < 10 else ""
        print(f"{lb:<9}{len(sub):>6}{hits:>5}{mhr*100:>9.1f}%{mkt*100:>8.1f}%{(mhr-mkt)*100:>+7.1f}{hold}")

    print("\n=== サニティ&控除率感度: 市場probの絶対値(上位3組で35-50%目安) と 控除率別超過符号 ===")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        sub = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        if len(sub) == 0 or int(sub.hit.sum()) < 10:
            print(f"  {lb:<9} 保留(的中<10)")
            continue
        mhr = sub.hit.mean()
        norm = sub.norm_implied.mean()
        signs = ["＋" if mhr - sub[f"imp_{int(t*100)}"].mean() > 0 else "－" for t in TAKEOUTS]
        signn = "＋" if mhr - norm > 0 else "－"
        stable = "符号不変" if len(set(signs + [signn])) == 1 else "符号が動く=判定不能"
        print(f"  {lb:<9} 正規化prob{norm*100:4.1f}% / 控除率22-28%超過符号[{signn}主|{'/'.join(signs)}] {stable}")

    print("\n=== 従(控除率フリー): モデル上位3組 vs 市場鉄板3組 の的中率＋重複率 ===")
    print(f"{'本命単勝帯':<9}{'モデル的中':>10}{'市場鉄板3':>10}{'差':>7}{'重複(組数)':>10}")
    for (lo, hi), lb in zip(BANDS, BAND_LBL):
        sub = d[(d.tansho_odds >= lo) & (d.tansho_odds < hi)]
        if len(sub) == 0 or int(sub.hit.sum()) < 10:
            continue
        print(f"{lb:<9}{sub.hit.mean()*100:>9.1f}%{sub.market_fav3_hit.mean()*100:>9.1f}%"
              f"{(sub.hit.mean()-sub.market_fav3_hit.mean())*100:>+6.1f}{sub.overlap.mean():>9.2f}/3")
    print("\n※主=控除率フリー(正規化)。市場probが35-50%から大きく外れる帯は超過を割引。")
    print("※主と従が同符号→堅い。重複が高い(2-3/3)と従は無情報。的中<10は帯保留。")


if __name__ == "__main__":
    main()
