"""空データ/鮮度アサーション。GH Actions の daily-scan 内で scan の直後に走らせ、
「HTTPは通ったがデータ0件（IPブロック/パース破損）」や「古いキャッシュのまま」を
非ゼロ終了で Actions の失敗通知に乗せる。

判別軸は picks 数でなく venues_checked（開催場数）。
  競艇は毎日どこかで開催 → 0場はほぼ確実に取得失敗であって「正常な見送り日」ではない。
  picks=0 は全レース本命イン＝正常な見送りでも起きるので死活指標に使えない。

チェック:
  1. today_picks.json が存在
  2. date が今日(JST。TZ=Asia/Tokyo 前提)と一致 = 鮮度
  3. venues_checked >= MIN_VENUES = 取得が生きている

例:
  python -m src.healthcheck            # 今日を検証、失敗で exit 1
  python -m src.healthcheck --date 20260708
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from .scan import load_cache

MIN_VENUES = 1   # 競艇は毎日開催。0場=取得失敗と断定してよい保守閾値


def check(expect_date: str) -> list[str]:
    """失敗理由のリストを返す（空なら健全）。"""
    errs = []
    cache = load_cache()
    if cache is None:
        return ["today_picks.json が存在しない（scan未実行 or 書込失敗）"]
    if cache.get("date") != expect_date:
        errs.append(f"日付が古い: cache={cache.get('date')} expected={expect_date}（cronスキップ/settle鮮度）")
    v = cache.get("venues_checked")
    if v is None:
        errs.append("venues_checked キー無し（旧フォーマット/scan途中終了）")
    elif v < MIN_VENUES:
        errs.append(f"開催場 {v} < {MIN_VENUES} = 取得0件（IPブロック/パース破損の疑い）")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    args = ap.parse_args()
    errs = check(args.date)
    if errs:
        print("[FAIL] HEALTHCHECK FAILED:")
        for e in errs:
            print(f"   - {e}")
        return 1
    cache = load_cache()
    print(f"[OK] healthcheck OK: date={cache.get('date')} "
          f"venues={cache.get('venues_checked')} races={cache.get('races_scanned')} "
          f"picks={len(cache.get('picks', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
