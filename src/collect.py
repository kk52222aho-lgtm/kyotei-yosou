"""データ収集 CLI。

指定した期間・場・レースの出走表＋結果をスクレイピングして DB に蓄積する。

例:
  python -m src.collect --start 20260620 --end 20260629
  python -m src.collect --start 20260629 --end 20260629 --venues 07 08 09
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

from . import scraper, storage
from .venues import ALL_JCD, name as venue_name


def daterange(start: str, end: str):
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser(description="boatrace.jp からレースデータを収集")
    ap.add_argument("--start", required=True, help="開始日 YYYYMMDD")
    ap.add_argument("--end", required=True, help="終了日 YYYYMMDD")
    ap.add_argument("--venues", nargs="*", default=ALL_JCD, help="場コード(複数可)")
    ap.add_argument("--races", type=int, default=12, help="1日あたり最大レース数")
    ap.add_argument("--skip-existing", action="store_true",
                    help="結果まで保存済みのレースはスキップ")
    args = ap.parse_args()

    conn = storage.connect()
    total_races = 0
    total_rows = 0

    for date in daterange(args.start, args.end):
        for jcd in args.venues:
            jcd = str(jcd).zfill(2)
            empty_streak = 0
            for rno in range(1, args.races + 1):
                if args.skip_existing and storage.has_race(conn, date, jcd, rno):
                    continue
                entries = scraper.fetch_racelist(date, jcd, rno)
                time.sleep(scraper.SLEEP_SEC)
                if not entries:
                    empty_streak += 1
                    # 同じ場で連続して出走表が無ければ非開催とみなし打ち切り
                    if empty_streak >= 2:
                        break
                    continue
                empty_streak = 0
                before = scraper.fetch_beforeinfo(date, jcd, rno)
                time.sleep(scraper.SLEEP_SEC)
                finish = scraper.fetch_result(date, jcd, rno)
                time.sleep(scraper.SLEEP_SEC)
                rows = storage.save_race(conn, date, jcd, rno, entries, finish, before)
                total_races += 1
                total_rows += rows
                flags = []
                if not before: flags.append("直前なし")
                if not finish: flags.append("結果なし")
                flag = f" ({'/'.join(flags)})" if flags else ""
                print(f"  {date} {venue_name(jcd)}({jcd}) {rno}R  {rows}艇{flag}")

    conn.close()
    print(f"\n完了: {total_races} レース / {total_rows} 行を保存")


if __name__ == "__main__":
    main()
