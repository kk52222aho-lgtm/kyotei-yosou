"""SQLite による収集データの保存。

1艇1行の非正規化テーブル `entries` に出走表＋結果をまとめて格納する。
学習用にそのまま pandas へ読み込める形。
"""
from __future__ import annotations

import os
import sqlite3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "kyotei.db")

COLUMNS = [
    "lane", "reg", "racer_class", "name", "age", "weight",
    "f_count", "l_count", "avg_st",
    "nat_win", "nat_2rate", "nat_3rate",
    "loc_win", "loc_2rate", "loc_3rate",
    "motor_2rate", "motor_3rate", "boat_2rate", "boat_3rate",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    date        TEXT NOT NULL,
    jcd         TEXT NOT NULL,
    rno         INTEGER NOT NULL,
    lane        INTEGER NOT NULL,
    reg         TEXT,
    racer_class TEXT,
    name        TEXT,
    age         REAL,
    weight      REAL,
    f_count     INTEGER,
    l_count     INTEGER,
    avg_st      REAL,
    nat_win     REAL,
    nat_2rate   REAL,
    nat_3rate   REAL,
    loc_win     REAL,
    loc_2rate   REAL,
    loc_3rate   REAL,
    motor_2rate REAL,
    motor_3rate REAL,
    boat_2rate  REAL,
    boat_3rate  REAL,
    tenji_time   REAL,
    tilt         REAL,
    weight_today REAL,
    tenji_st     REAL,
    wind_speed   REAL,
    wave_height  REAL,
    temperature  REAL,
    water_temp   REAL,
    finish      INTEGER,
    win         INTEGER,
    PRIMARY KEY (date, jcd, rno, lane)
);
"""

# 直前情報のカラム（既存DBには ALTER で追加）
BEFORE_LANE_COLS = ["tenji_time", "tilt", "weight_today", "tenji_st"]
BEFORE_RACE_COLS = ["wind_speed", "wave_height", "temperature", "water_temp"]
BEFORE_COLS = BEFORE_LANE_COLS + BEFORE_RACE_COLS


def _migrate(conn) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(entries)")}
    for col in BEFORE_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE entries ADD COLUMN {col} REAL")
    conn.commit()


PAYOUT_SCHEMA = """
CREATE TABLE IF NOT EXISTS payouts (
    date TEXT NOT NULL, jcd TEXT NOT NULL, rno INTEGER NOT NULL,
    trifecta_combo TEXT, trifecta_yen INTEGER,
    trio_combo TEXT, trio_yen INTEGER,
    exacta_combo TEXT, exacta_yen INTEGER,
    quinella_combo TEXT, quinella_yen INTEGER,
    tansho_lane INTEGER, tansho_yen INTEGER,
    PRIMARY KEY (date, jcd, rno)
);
"""

PAYOUT_COLS = ["trifecta_combo", "trifecta_yen", "trio_combo", "trio_yen",
               "exacta_combo", "exacta_yen", "quinella_combo", "quinella_yen",
               "tansho_lane", "tansho_yen"]


def connect() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.execute(PAYOUT_SCHEMA)
    _migrate(conn)
    return conn


def save_payouts(conn, date: str, jcd: str, rno: int, p: dict) -> None:
    vals = [date, str(jcd).zfill(2), rno] + [p.get(c) for c in PAYOUT_COLS]
    ph = ",".join(["?"] * len(vals))
    cols = "date,jcd,rno," + ",".join(PAYOUT_COLS)
    conn.execute(f"INSERT OR REPLACE INTO payouts ({cols}) VALUES ({ph})", vals)


def save_race(conn, date: str, jcd: str, rno: int,
              entries: list[dict], finish: dict[int, int] | None,
              before: tuple[dict, dict] | None = None) -> int:
    """1レース分(6艇)を upsert。finish=着順、before=直前情報があれば付与。"""
    per_lane, race_cond = before if before else ({}, {})
    all_cols = COLUMNS + BEFORE_COLS
    rows = 0
    for e in entries:
        lane = e["lane"]
        f = finish.get(lane) if finish else None
        win = 1 if f == 1 else (0 if f is not None else None)
        bl = per_lane.get(lane, {})
        rec = dict(e)
        for c in BEFORE_LANE_COLS:
            rec[c] = bl.get(c)
        for c in BEFORE_RACE_COLS:
            rec[c] = race_cond.get(c)
        vals = [date, str(jcd).zfill(2), rno] + [rec.get(c) for c in all_cols] + [f, win]
        placeholders = ",".join(["?"] * len(vals))
        cols = "date,jcd,rno," + ",".join(all_cols) + ",finish,win"
        conn.execute(
            f"INSERT OR REPLACE INTO entries ({cols}) VALUES ({placeholders})", vals
        )
        rows += 1
    conn.commit()
    return rows


def has_race(conn, date: str, jcd: str, rno: int) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM entries WHERE date=? AND jcd=? AND rno=? AND finish IS NOT NULL LIMIT 1",
        (date, str(jcd).zfill(2), rno),
    )
    return cur.fetchone() is not None
