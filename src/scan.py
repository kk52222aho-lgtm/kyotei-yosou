"""本日の妙味レース・スキャナ。

検証済みの「勝てる形」（モデル本命≠1号艇）に該当するレースを全開催場から抽出し、
推奨買い目（単勝・本命／2連単上位3点）を一覧表示する。日々の実戦ツール。

注意: 出走表ベースで予測（展示タイム等の直前情報は未発表なら欠損補完）。
本命≠1号艇のレースだけ単勝オッズを取得して表示。

例:
  python -m src.scan                 # 本日
  python -m src.scan --date 20260630
  python -m src.scan --no-odds       # オッズ取得を省きさらに高速
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

from . import predict, scraper, storage
from .venues import name as venue_name

CACHE_PATH = os.path.join(storage.DATA_DIR, "today_picks.json")


def save_cache(date: str, picks: list[dict]) -> None:
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    payload = {"date": date,
               "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
               "picks": picks}
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def load_cache() -> dict | None:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def scan(date: str, with_odds: bool = True, bundle=None):
    bundle = bundle or predict.load_model()
    if bundle is None:
        print("モデル未学習です。python -m src.train を先に。")
        return []
    venues = scraper.fetch_held_venues(date)
    print(f"{date} 開催 {len(venues)}場: {' '.join(venue_name(j) for j in venues)}")
    print("スキャン中…（本命≠1号艇のレースを抽出）\n")

    picks = []
    for jcd in venues:
        deadlines = scraper.fetch_deadlines(date, jcd)  # {rno: "HH:MM"} 場ごと1発
        empty = 0
        for rno in range(1, 13):
            entries = scraper.fetch_racelist(date, jcd, rno)
            time.sleep(0.4)  # サーバ配慮
            if not entries:
                empty += 1
                if empty >= 2:
                    break
                continue
            empty = 0
            for e in entries:
                e["jcd"] = jcd
            rows = predict.predict_entries(entries, bundle)
            rec = predict.recommend(rows)
            if not rec.get("bet"):
                continue
            odds = None
            if with_odds:
                o = scraper.fetch_odds(date, jcd, rno)
                odds = o.get(rec["tansho"]) if o else None
            picks.append({
                "jcd": jcd, "venue": venue_name(jcd), "rno": rno,
                "honmei": rec["tansho"], "name": rec.get("tansho_name"),
                "win_pct": rows[0]["win_pct"], "exacta3": rec["exacta3"],
                "odds": odds, "deadline": deadlines.get(rno),
            })
            od = f"{odds:.1f}倍" if odds else "—"
            dl = deadlines.get(rno)
            print(f"  ⚑ {venue_name(jcd)} {rno}R {('締切'+dl) if dl else ''}  単勝{rec['tansho']}号 "
                  f"{rec.get('tansho_name','')} (予想{rows[0]['win_pct']}% / {od})  2連単{'・'.join(rec['exacta3'])}")
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--no-odds", action="store_true")
    args = ap.parse_args()
    picks = scan(args.date, with_odds=not args.no_odds)
    save_cache(args.date, picks)
    print(f"\n=== 本日の妙味レース {len(picks)}件 ===  (キャッシュ保存: {CACHE_PATH})")
    if not picks:
        print("該当なし（全レースでモデル本命がイン＝見送り）。")


if __name__ == "__main__":
    main()
