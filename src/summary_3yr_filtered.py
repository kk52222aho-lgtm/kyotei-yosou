"""現行戦略フル3年・単勝<1.5丸ごと見送りフィルタ込みの正確な成績。

本命の単勝オッズ:
  - 本命が勝ち → tansho_yen/100（データから正確）
  - 本命が負け → oddstf をスクレイプ（負けレースのみ・全体の約半分）
フィルタ: 本命単勝オッズ<1.5 のレースは丸ごと見送り（単勝も2連単も）。

例:
  python -m src.summary_3yr_filtered
"""
from __future__ import annotations

import time

import pandas as pd

from . import scraper
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS

FLOOR = 1.5


def collect(df):
    races = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["tansho_lane"]):
                continue
            s = g["p"].sum()
            top3 = []
            if s > 0 and not pd.isna(top["exacta_yen"]):
                p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
                top3 = [c for c, _ in exacta_probs(p)[:3]]
            won = int(top["tansho_lane"]) == int(top["lane"])
            races.append({
                "year": y, "date": date, "jcd": jcd, "rno": int(rno),
                "honmei": int(top["lane"]), "won": won,
                "tansho_yen": int(top["tansho_yen"]),
                "top3": top3,
                "exacta_combo": top["exacta_combo"],
                "exacta_yen": None if pd.isna(top["exacta_yen"]) else int(top["exacta_yen"]),
            })
    return races


def main():
    df = load()
    races = collect(df)
    losers = [r for r in races if not r["won"]]
    print(f"妙味レース {len(races)} / うち本命負け {len(losers)}（この分だけオッズ取得）")

    for i, r in enumerate(losers):
        od = scraper.fetch_odds(r["date"], r["jcd"], r["rno"])
        r["_odds"] = od.get(r["honmei"]) if od else None
        time.sleep(0.3)
        if (i + 1) % 200 == 0:
            print(f"  odds {i+1}/{len(losers)}")

    for r in races:
        if r["won"]:
            r["odds"] = r["tansho_yen"] / 100.0
        else:
            r["odds"] = r.get("_odds")  # None なら不明

    def stats(rows):
        # 単勝
        t_n = len(rows)
        t_hit = sum(1 for r in rows if r["won"])
        t_ret = sum(r["tansho_yen"] for r in rows if r["won"])
        # 2連単(3点、top3ありのみ)
        er = [r for r in rows if r["top3"] and r["exacta_yen"] is not None]
        e_n = len(er)
        e_hit = sum(1 for r in er if r["exacta_combo"] in r["top3"])
        e_ret = sum(r["exacta_yen"] for r in er if r["exacta_combo"] in r["top3"])
        stake = t_n * 100 + e_n * 3 * 100
        ret = t_ret + e_ret
        return {"races": t_n, "t_hit": t_hit, "t_ret": t_ret, "t_roi": t_ret/(t_n*100) if t_n else 0,
                "e_n": e_n, "e_hit": e_hit, "e_roi": e_ret/(e_n*300) if e_n else 0,
                "roi": ret/stake if stake else 0, "pl": ret - stake, "stake": stake}

    allr = stats(races)
    # フィルタ: オッズ不明は「切れないので残す」(保守的)。<1.5のみ除外
    kept = [r for r in races if not (r["odds"] is not None and r["odds"] < FLOOR)]
    skipped = len(races) - len(kept)
    filt = stats(kept)

    def show(label, s):
        print(f"{label:<16}勝負{s['races']:>5} 単勝{s['t_hit']/s['races']*100:4.1f}%/{s['t_roi']*100:5.1f}% "
              f"2連単{s['e_hit']/s['e_n']*100:4.1f}%/{s['e_roi']*100:5.1f}% "
              f"合計{s['roi']*100:5.1f}% 収支{s['pl']:+,}円")

    print("\n=== フル3年(2023-25) 現行戦略 ===")
    show("フィルタ無し", allr)
    show(f"単勝<1.5見送り", filt)
    print(f"\n見送り {skipped}レース（本命単勝<1.5）。"
          f"オッズ不明の負け {sum(1 for r in losers if r.get('_odds') is None)}件は保守的に残した。")


if __name__ == "__main__":
    main()
