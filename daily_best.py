"""外部オーケストレータ向け: 本日の最良EV単勝ピックを1本だけJSONで返すスタンドアロンCLI。

pick   … 本日(JST)の開催全場を走査し、締切が近い(窓内の)レースを対象に
         EV(=校正済みwin_prob×単勝オッズ) >= 0.90 を満たす艇の中から
         勝率最大の1点を出す(「一番当たりそうでEVが土俵に乗っている」買い目)。
         進捗はstderr、結果はstdoutの最終1行(JSON)のみ。
settle … 指定レースの公式結果から win/lose と単勝払戻(100円あたり)を返す。

例:
  python daily_best.py pick
  python daily_best.py pick --date 20260729
  python daily_best.py settle --date 20260728 --jcd 12 --race 1 --selection 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

from src import predict, scraper
from src.venues import name as venue_name

JST = dt.timezone(dt.timedelta(hours=9), "JST")

# 選定基準: EV >= MIN_EV を満たす中で勝率(win_prob)最大の1点。
# 「一番当たりそうで、かつ期待値が土俵に乗っている」買い目を出す。
MIN_EV = 0.90
# 閑散プールの未形成オッズがEVを爆発させる罠への保険(0で無効)
MIN_PROB = 0.03
MAX_ODDS = 30.0
# 締切がこの分数以内のレースのみ対象。遠い未来の締切は朝イチ暫定オッズで
# EVが蜃気楼になる(odds_cache.jsonの教訓)ため窓で絞る(0で無効)
WINDOW_MIN = 90.0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _deadline_dt(date: str, hhmm: str) -> dt.datetime | None:
    """'HH:MM' + YYYYMMDD → JSTのdatetime。パース不能はNone。"""
    try:
        h, m = map(int, hhmm.split(":"))
        d = dt.datetime.strptime(date, "%Y%m%d")
        return dt.datetime(d.year, d.month, d.day, h, m, tzinfo=JST)
    except (ValueError, AttributeError):
        return None


def pick(date: str, min_prob: float = MIN_PROB, max_odds: float = MAX_ODDS,
         window_min: float = WINDOW_MIN, min_ev: float = MIN_EV) -> dict:
    bundle = predict.load_model()
    if bundle is None:
        return {"pick": None, "reason": "モデル未学習 (data/model.joblib なし)"}
    now = dt.datetime.now(JST)
    venues = scraper.fetch_held_venues(date)
    log(f"{date} 開催 {len(venues)}場: {' '.join(venue_name(j) for j in venues)}")
    if not venues:
        return {"pick": None, "reason": f"{date} は開催なし"}

    best = None
    open_races = 0
    for jcd in venues:
        deadlines = scraper.fetch_deadlines(date, jcd)  # {rno: "HH:MM"} 場ごと1発
        time.sleep(0.4)  # サーバ配慮
        if not deadlines:
            log(f"  {venue_name(jcd)}: 締切情報なし→場ごとスキップ")
            continue
        empty = 0
        for rno in range(1, 13):
            dl = deadlines.get(rno)
            dl_dt = _deadline_dt(date, dl) if dl else None
            if dl_dt is None or dl_dt <= now:
                continue  # 締切不明 or 締切済み
            if window_min > 0 and dl_dt > now + dt.timedelta(minutes=window_min):
                continue  # 締切が遠くオッズ未形成
            entries = scraper.fetch_racelist(date, jcd, rno)
            time.sleep(0.4)  # サーバ配慮
            if not entries:
                empty += 1
                if empty >= 2:
                    break
                continue
            empty = 0
            open_races += 1
            for e in entries:
                e["jcd"] = jcd
            rows = predict.predict_entries(entries, bundle)
            odds = scraper.fetch_odds(date, jcd, rno)
            time.sleep(0.4)  # サーバ配慮
            if not odds:
                log(f"  {venue_name(jcd)} {rno}R 締切{dl}: オッズ未取得→スキップ")
                continue
            rows = predict.attach_odds(rows, odds)
            top = max(
                (r for r in rows
                 if r.get("ev") is not None and r["ev"] >= min_ev
                 and (not min_prob or r["win_prob"] >= min_prob)
                 and (not max_odds or r["odds"] <= max_odds)),
                key=lambda r: r["win_prob"], default=None)
            if top is None:
                continue
            log(f"  {venue_name(jcd)} {rno}R 締切{dl}: best {top['lane']}号 "
                f"p={top['win_prob']:.3f} x {top['odds']:.1f}倍 = EV {top['ev']:.2f}")
            if best is None or top["win_prob"] > best["model_p"]:
                best = {
                    "genre": "競艇",
                    "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                    "jcd": jcd,
                    "venue": venue_name(jcd),
                    "race_no": rno,
                    "bet_type": "単勝",
                    "selection": str(top["lane"]),
                    "model_p": top["win_prob"],
                    "odds": top["odds"],
                    "ev": top["ev"],
                    "deadline": dl,
                }
    if best is None:
        return {"pick": None,
                "reason": f"締切前のレースなし（開催{len(venues)}場/走査{open_races}R）"}
    return best


def settle(date: str, jcd: str, rno: int, selection: str) -> dict:
    res = scraper.fetch_result_full(date, jcd, rno)
    if res is None or res.get("winner") is None:
        return {"result": "unknown", "payout": 0}
    if str(res["winner"]) != str(selection):
        return {"result": "lose", "payout": 0}
    yen = res.get("tansho_yen")
    if yen is None:
        # 結果ページに払戻が無い場合の保険: 確定オッズ×100円
        odds = scraper.fetch_odds(date, jcd, rno)
        o = odds.get(int(selection)) if odds else None
        yen = int(round(o * 100)) if o else 0
    return {"result": "win", "payout": yen}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pick", help="本日の最良EV単勝を1本返す")
    p1.add_argument("--date", default=dt.datetime.now(JST).strftime("%Y%m%d"))
    p1.add_argument("--min-prob", type=float, default=MIN_PROB)
    p1.add_argument("--max-odds", type=float, default=MAX_ODDS)
    p1.add_argument("--window-min", type=float, default=WINDOW_MIN)
    p1.add_argument("--min-ev", type=float, default=MIN_EV)
    p2 = sub.add_parser("settle", help="指定レースの単勝結果を返す")
    p2.add_argument("--date", required=True)
    p2.add_argument("--jcd", required=True)
    p2.add_argument("--race", type=int, required=True)
    p2.add_argument("--selection", required=True)
    args = ap.parse_args()

    if args.cmd == "pick":
        out = pick(args.date, args.min_prob, args.max_odds, args.window_min,
                   args.min_ev)
    else:
        out = settle(args.date, args.jcd, args.race, args.selection)
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
