# -*- coding: utf-8 -*-
"""古うなった派生物を作り直す。**毎朝の輪に積むためのもん。**

status_beacon が「model.joblib が56日STALE」と毎朝鳴っとったのに56日直らんかった
(2026-09-11に発覚)。鳴っとるのに直らんのは、**鳴らす側と直す側が繋がっとらん**から。
収集器が在るのに登録されとらんかった src.official / oriten と同じ穴や。

毎日全部やり直すと無駄なんで、**年齢を見て古い時だけ**回す。
閾値は status_beacon の許容より短うしとく(鳴る前に直す)。

    python -m src.refresh_derived            # 古いもんだけ
    python -m src.refresh_derived --force    # 全部
    python -m src.refresh_derived --dry-run  # 何をやるかだけ
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

from . import storage

# (ファイル名, これより古かったら作り直す日数, 呼ぶモジュール, 説明)
JOBS = [
    ("model.joblib", 7, "src.train", "本番モデル(status_beacon の許容は14日)"),
    ("motor_boat.csv", 30, "src.backfill_motor_boat", "モーター/ボート番号(許容60日)"),
]

# 🚨 2026-09-24: **同じ穴の4件目**。status_beacon は `=> STALE: kachimake` を
# 17日鳴らし続けとったのに、`src/build_kachimake.py` を呼ぶもんがどこにも
# 無かった(src.official 27日 / oriten 54日 / src.train 56日 と同じ形)。
# 派生物がファイルやのうて**DBの表**やから、年齢はファイルの mtime やのうて
# 表の中の最新日で測る。
TABLE_JOBS = [
    # 閾値は 3 やのうて 7。build_kachimake は差分やのうて **entries 全件(159万行)を
    # 毎回作り直す**んで、3日にしたら実質毎朝フル再構築になる。
    # beacon の許容 14日の内側に収まっとったらええ。
    ("kachimake", 7, "src.build_kachimake", "勝ち負け(全件再構築。status_beacon の許容は14日)"),
]


def table_lag_days(table: str) -> float | None:
    """表の中の最新日が何日前か。表が無い/空なら None。"""
    try:
        conn = storage.connect()
        row = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()
        conn.close()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    d = dt.datetime.strptime(str(row[0]), "%Y%m%d").date()
    return (dt.date.today() - d).days


def age_days(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    return (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 86400


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rc = 0
    for name, limit, mod, note in JOBS:
        path = os.path.join(storage.DATA_DIR, name)
        age = age_days(path)
        if age is None:
            why = "無い"
        elif age > limit:
            why = f"{age:.0f}日 > {limit}日"
        elif a.force:
            why = f"{age:.0f}日 (--force)"
        else:
            print(f"[skip] {name}: {age:.0f}日 <= {limit}日  {note}", flush=True)
            continue
        print(f"[作り直す] {name}: {why}  -> {mod}   {note}", flush=True)
        if a.dry_run:
            continue
        before = age_days(path)
        try:
            __import__(mod, fromlist=["main"]).main()
        except Exception as ex:
            rc = 1
            print(f"[NG] {name}: {type(ex).__name__}: {ex}", flush=True)
            continue
        after = age_days(path)
        # 🚨 「走った」と「新しなった」は別。実物の年齢で確かめる
        if after is None or (before is not None and after >= before):
            rc = 1
            print(f"[NG] {name}: 呼んだが新しなっとらん (前 {before} 日 / 後 {after} 日)", flush=True)
        else:
            print(f"[OK] {name}: {after*24:.1f} 時間前の版になった", flush=True)

    for table, limit, mod, note in TABLE_JOBS:
        lag = table_lag_days(table)
        if lag is None:
            why = "表が空か無い"
        elif lag > limit:
            why = f"最新が {lag:.0f}日前 > {limit}日"
        elif a.force:
            why = f"最新が {lag:.0f}日前 (--force)"
        else:
            print(f"[skip] {table}: 最新が {lag:.0f}日前 <= {limit}日  {note}", flush=True)
            continue
        print(f"[作り直す] {table}: {why}  -> {mod}   {note}", flush=True)
        if a.dry_run:
            continue
        before = lag
        try:
            __import__(mod, fromlist=["main"]).main()
        except Exception as ex:
            rc = 1
            print(f"[NG] {table}: {type(ex).__name__}: {ex}", flush=True)
            continue
        after = table_lag_days(table)
        # 🚨 ファイルの時と同じで、「走った」と「新しなった」は別
        if after is None or (before is not None and after >= before):
            rc = 1
            print(f"[NG] {table}: 呼んだが新しなっとらん "
                  f"(前 {before}日前 / 後 {after}日前)", flush=True)
        else:
            print(f"[OK] {table}: 最新が {after}日前になった", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
