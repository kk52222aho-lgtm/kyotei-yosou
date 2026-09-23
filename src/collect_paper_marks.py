"""宮島の紙から**記者の予想印**(◎○▲△)を抜く。

`collect_paper.py` は選手コメントだけ取って、**印を捨てとった**。
印は「予想屋の本命そのもの」やから、盤と並べたら
「プロの予想屋は盤より上か」が直接測れる。

## 位置

    x ≈ 50 の1列に、艇番の並び順で ◎○▲△ が入る(5・6号艇は無印のことが多い)
    x ≈ 383/397/411 の3列は別の格子(正体を特定でけとらんので取らん)

印は1文字なんで pdfminer の**文字単位**の座標で拾う。
行の対応は登録番号の箱の上端に**いちばん近い**もんへ(行間19ptなんで±10で一意)。

## 取らんもん

結果と払戻。紙に刷られとるが、`collect_paper.py` と同じく**列として持たん**。
"""
from __future__ import annotations

import argparse
import io
import re
import sqlite3
import time
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTTextContainer

from .collect_paper import JCD, REG_RE, BARE_REG_RE, X_REG, board_races

X_MARK = (40.0, 62.0)
MARKS = "◎○▲△"
ROW_TOL = 10.0

DDL = """
CREATE TABLE IF NOT EXISTS paper_mark (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER, reg TEXT,
  mark TEXT, fetched_at TEXT,
  PRIMARY KEY (date, jcd, rno, lane)
);
CREATE TABLE IF NOT EXISTS paper_mark_done (
  date TEXT, jcd TEXT, status TEXT, n_race INTEGER, n_mark INTEGER,
  note TEXT, fetched_at TEXT,
  PRIMARY KEY (date, jcd)
);
"""


def _chars(page) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []

    def walk(o):
        for e in o:
            if isinstance(e, LTChar):
                t = e.get_text().strip()
                if t in MARKS and X_MARK[0] <= e.bbox[0] <= X_MARK[1]:
                    out.append((e.bbox[0], e.bbox[3], t))
            elif hasattr(e, "__iter__"):
                walk(e)

    walk(page)
    return out


def parse_marks(b: bytes) -> list[list[tuple[str, str]]]:
    """[レース塊][(登録番号, 印)] を返す。印が無い艇は ''。"""
    laparams = LAParams(detect_vertical=True, line_margin=0.3)
    races: list[list[tuple[str, str]]] = []
    for page in extract_pages(io.BytesIO(b), laparams=laparams):
        items = [e for e in page if isinstance(e, LTTextContainer)]
        regs: list[tuple[float, str]] = []
        for e in items:
            if X_REG[0] <= e.bbox[0] <= X_REG[1]:
                t = e.get_text().strip()
                m = REG_RE.search(t)
                if m:
                    regs.append((e.bbox[3], m.group(2)))
                elif BARE_REG_RE.fullmatch(t):
                    regs.append((e.bbox[3], t.strip()))
        regs.sort(key=lambda z: -z[0])
        ms = _chars(page)
        for i in range(0, len(regs) - 5, 6):
            chunk = regs[i : i + 6]
            gaps = [chunk[k][0] - chunk[k + 1][0] for k in range(5)]
            if max(gaps) > 3 * (min(gaps) or 1):
                continue
            rows: list[tuple[str, str]] = []
            for y, reg in chunk:
                near = [(abs(my - y), mk) for _, my, mk in ms if abs(my - y) <= ROW_TOL]
                near.sort()
                rows.append((reg, near[0][1] if near else ""))
            races.append(rows)
    return races


def collect_day(con: sqlite3.Connection, date: str, cache: Path) -> str:
    p = cache / f"{date}.pdf"
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not p.exists():
        con.execute("INSERT OR REPLACE INTO paper_mark_done VALUES (?,?,?,?,?,?,?)",
                    (date, JCD, "no_pdf", 0, 0, "", now))
        return "no_pdf"
    races = parse_marks(p.read_bytes())
    want = board_races(con, date)
    n_race = n_mark = 0
    bad_honmei = 0
    for rows in races:
        key = frozenset(r for r, _ in rows)
        if key not in want:
            continue
        rno, lane_of = want[key]
        n_race += 1
        if sum(1 for _, mk in rows if mk == "◎") != 1:
            bad_honmei += 1        # 本命が1つやない = **数えて残す**。黙って捨てん
        for reg, mk in rows:
            if not mk:
                continue
            n_mark += 1
            con.execute("INSERT OR REPLACE INTO paper_mark VALUES (?,?,?,?,?,?,?)",
                        (date, JCD, rno, lane_of[reg], reg, mk, now))
    status = "ok" if n_race else "no_match"
    con.execute("INSERT OR REPLACE INTO paper_mark_done VALUES (?,?,?,?,?,?,?)",
                (date, JCD, status, n_race, n_mark, f"本命が1つでない塊={bad_honmei}", now))
    return f"{status} {n_race}R/{n_mark}印 本命異常={bad_honmei}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cache", default="data/paper/17")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(DDL)
    con.commit()
    days = [r[0] for r in con.execute(
        "SELECT date FROM paper_done WHERE jcd=? AND status='ok' ORDER BY date", (JCD,))]
    if not a.redo:
        done = {r[0] for r in con.execute(
            "SELECT date FROM paper_mark_done WHERE jcd=? AND status='ok'", (JCD,))}
        days = [d for d in days if d not in done]
    if a.limit:
        days = days[: a.limit]
    print(f"{len(days)} 日を読む(通信なし。キャッシュのPDFだけ)")
    for i, d in enumerate(days, 1):
        r = collect_day(con, d, Path(a.cache))
        con.commit()
        if i % 25 == 0 or i == len(days):
            print(f"[{i}/{len(days)}] {d} -> {r}", flush=True)
    n = con.execute("SELECT COUNT(*) FROM paper_mark").fetchone()[0]
    print(f"\npaper_mark {n:,} 行")
    for s, c in con.execute("SELECT status,COUNT(*) FROM paper_mark_done GROUP BY status"):
        print(f"  {s}: {c} 日")
    print("  印の内訳:", dict(con.execute("SELECT mark,COUNT(*) FROM paper_mark GROUP BY mark")))


if __name__ == "__main__":
    main()
