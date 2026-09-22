"""宮島の「デイリーPDF新聞」から選手コメントを退避する。

    https://www.boatrace-miyajima.com/yosousi/dailysport.php?mode=1&date=YYYYMMDD&save=1

24場のうち、過去日付を日付指定で引けるんはここだけ(2026-09-22 の走査)。
他場の共通CMS(/modules/raceinfo/?page=index_racers_comment&targetday=...)は
targetday を受け取って黙って捨てる(4日付で md5 が1個)。

## 取るもん / 取らんもん

取る  : 選手コメント(場が採った一次コメント)・登録番号
取らん: 結果と払戻。**同じ紙に刷られとる**ので、列として持たんことで先読みを構造的に断つ。
        成績の数値も取らん(盤の entries に同じもんが在る。360/360 で一致を確認済み)。

## 凍っとることの確認(2026-09-22 → 09-23)

同じ日(20260922)の紙を 11:47 / 11:55 / 翌朝 の3回撮って文字単位で差分を取った:

    差分ブロック 352本 … 全部が数値。日本語 0本 / 印◎○▲△ 0本 / 英字 0本
    印の総数 264 → 264

= **コメント・記者の見解・印は当日中も日をまたいでも1文字も動かん。**
動くんは結果と、それに連動する数値だけ。

## 位置決め

平坦化したテキストやのうて座標で取る。
  行 = 箱の**上端** y1(登録番号の箱 x≈79 と コメントの箱 x≈686 が同じ上端に並ぶ)
  列 = x

🚨 下端 y0 で合わせたらあかん。**1行の箱と2行の箱で下端が 8pt ずれる**(上端は揃う)。
下端でやった初版は 432 艇中 16 艇を取りこぼして、しかも紙の側は空白やなかった。
上端に変えたら ±2pt で 30/30、複数当たりゼロ。
レース番号は**位置で決めん**。6人ぶんの登録番号の集合が entries のどの rno と
一致するかで決める。一致せんかったら書かずに落とす。
→ [[insight_anchor_found_span_unmeasured]] / [[insight_query_label_is_not_a_match]]
"""
from __future__ import annotations

import argparse
import io
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer

JCD = "17"  # 宮島
URL = "https://www.boatrace-miyajima.com/yosousi/dailysport.php?mode=1&date={date}&save=1"
UA = "Mozilla/5.0 (compatible; kyotei-yosou research collector)"

# 非開催日とアーカイブ欠損は**同じから紙**で返る。1ページ目の文字数で分ける
# (実紙 5,600〜9,100字 / から紙 1,940字)。しきい値は実測の中間。
EMPTY_MAX_CHARS = 3000

X_REG = (70.0, 100.0)     # 「NN期\ndddd」の箱
X_COMMENT = (675.0, 815.0)  # 選手コメントの箱
Y_TOL = 4.0               # 上端どうしの差。実測は ±2 で足りる

REG_RE = re.compile(r"(\d{2,3})期\s*(\d{4})")
# 締切時刻の箱が同じ x の帯に入っとるので弾く
TIME_RE = re.compile(r"^\d{1,2}:\d{1,2}$")

DDL = """
CREATE TABLE IF NOT EXISTS paper_comment (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER, reg TEXT,
  comment TEXT, fetched_at TEXT,
  PRIMARY KEY (date, jcd, rno, lane)
);
CREATE TABLE IF NOT EXISTS paper_done (
  date TEXT, jcd TEXT, status TEXT, n_race INTEGER, n_comment INTEGER,
  n_reg_match INTEGER, note TEXT, fetched_at TEXT,
  PRIMARY KEY (date, jcd)
);
"""


def fetch_pdf(date: str, cache_dir: Path, force: bool = False) -> bytes | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{date}.pdf"
    if p.exists() and not force:
        return p.read_bytes()
    req = urllib.request.Request(URL.format(date=date), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            b = r.read()
    except Exception as e:  # noqa: BLE001
        print(f"  {date}: 取得失敗 {e}", file=sys.stderr)
        return None
    if not b.startswith(b"%PDF"):
        print(f"  {date}: PDF やない ({b[:16]!r})", file=sys.stderr)
        return None
    p.write_bytes(b)
    return b


def parse_pdf(b: bytes) -> tuple[list[list[tuple[str, str]]], int]:
    """PDF から [レース塊][(登録番号, コメント)] を返す。第2要素は1ページ目の文字数。"""
    laparams = LAParams(detect_vertical=True, line_margin=0.3)
    races: list[list[tuple[str, str]]] = []
    first_page_chars = 0
    for pi, page in enumerate(extract_pages(io.BytesIO(b), laparams=laparams)):
        items = [e for e in page if isinstance(e, LTTextContainer)]
        if pi == 0:
            first_page_chars = sum(len(e.get_text().strip()) for e in items)
        regs: list[tuple[float, str]] = []
        cmts: list[tuple[float, str]] = []
        for e in items:
            x0, y1 = e.bbox[0], e.bbox[3]  # y1 = 箱の上端
            t = e.get_text().strip()
            if X_REG[0] <= x0 <= X_REG[1]:
                m = REG_RE.search(t)
                if m:
                    regs.append((y1, m.group(2)))
            elif X_COMMENT[0] <= x0 <= X_COMMENT[1] and len(t) > 4:
                if TIME_RE.match(t):
                    continue
                # 折り返しの改行は詰める(紙の都合であって本文やない)
                cmts.append((y1, re.sub(r"\s+", "", t)))
        regs.sort(key=lambda z: -z[0])
        # 6本ずつがレース塊。端数は捨てる(塊が崩れとる紙は書かん)
        for i in range(0, len(regs) - 5, 6):
            chunk = regs[i : i + 6]
            # 塊の中の y の間隔から外れたら別の塊。間隔の3倍を境にする
            gaps = [chunk[k][0] - chunk[k + 1][0] for k in range(5)]
            if max(gaps) > 3 * (min(gaps) or 1):
                continue
            rows: list[tuple[str, str]] = []
            for y, reg in chunk:
                hit = sorted((cy, c) for cy, c in cmts if abs(cy - y) <= Y_TOL)
                # 実測では1艇1箱。割れとったら上から繋ぐ(黙って落とさん)
                rows.append((reg, "".join(c for _, c in reversed(hit))))
            races.append(rows)
    return races, first_page_chars


def board_races(con: sqlite3.Connection, date: str) -> dict[frozenset[str], tuple[int, dict[str, int]]]:
    """entries から {登録番号の集合: (rno, {reg: lane})} を作る。"""
    out: dict[frozenset[str], tuple[int, dict[str, int]]] = {}
    q = "SELECT rno, lane, reg FROM entries WHERE date=? AND jcd=? ORDER BY rno, lane"
    byrace: dict[int, dict[str, int]] = {}
    for rno, lane, reg in con.execute(q, (date, JCD)):
        byrace.setdefault(int(rno), {})[str(reg)] = int(lane)
    for rno, m in byrace.items():
        if len(m) == 6:
            out[frozenset(m)] = (rno, m)
    return out


def collect_day(con: sqlite3.Connection, date: str, cache: Path, force: bool) -> str:
    b = fetch_pdf(date, cache, force)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if b is None:
        con.execute(
            "INSERT OR REPLACE INTO paper_done VALUES (?,?,?,?,?,?,?,?)",
            (date, JCD, "fetch_failed", 0, 0, 0, "", now),
        )
        return "fetch_failed"
    races, chars = parse_pdf(b)
    if chars <= EMPTY_MAX_CHARS:
        # 非開催日 or アーカイブ欠損。**0件は測定結果や**ので status に残す
        con.execute(
            "INSERT OR REPLACE INTO paper_done VALUES (?,?,?,?,?,?,?,?)",
            (date, JCD, "blank", 0, 0, 0, f"1ページ目={chars}字", now),
        )
        return "blank"
    want = board_races(con, date)
    n_race = n_cmt = n_match = 0
    unresolved = 0
    for rows in races:
        key = frozenset(r for r, _ in rows)
        if key not in want:
            unresolved += 1
            continue
        rno, lane_of = want[key]
        n_race += 1
        n_match += 6
        for reg, comment in rows:
            if not comment:
                continue
            n_cmt += 1
            con.execute(
                "INSERT OR REPLACE INTO paper_comment VALUES (?,?,?,?,?,?,?)",
                (date, JCD, rno, lane_of[reg], reg, comment, now),
            )
    status = "ok" if unresolved == 0 and n_race else ("partial" if n_race else "no_match")
    con.execute(
        "INSERT OR REPLACE INTO paper_done VALUES (?,?,?,?,?,?,?,?)",
        (date, JCD, status, n_race, n_cmt, n_match, f"照合でけん塊={unresolved}", now),
    )
    return f"{status} {n_race}R/{n_cmt}コメント 照合でけん塊={unresolved}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cache", default="data/paper/17")
    ap.add_argument("--from", dest="dfrom", default=None, help="YYYYMMDD")
    ap.add_argument("--to", dest="dto", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--redo", action="store_true", help="済みの日もやり直す")
    a = ap.parse_args()

    con = sqlite3.connect(a.db, timeout=120)
    # 常駐(KyoteiDailyCollect/KyoteiVenueSnap)が同じ db を触る。鍵待ちを入れんと
    # 退避の途中で database is locked で死ぬ(2026-09-23 に実際に72日目で落ちた)
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(DDL)
    con.commit()

    q = "SELECT DISTINCT date FROM entries WHERE jcd=?"
    args: list = [JCD]
    if a.dfrom:
        q += " AND date>=?"
        args.append(a.dfrom)
    if a.dto:
        q += " AND date<=?"
        args.append(a.dto)
    q += " ORDER BY date"
    days = [r[0] for r in con.execute(q, args)]
    if not a.redo:
        done = {r[0] for r in con.execute("SELECT date FROM paper_done WHERE jcd=? AND status IN ('ok','blank')", (JCD,))}
        days = [d for d in days if d not in done]
    if a.limit:
        days = days[: a.limit]

    print(f"宮島の開催日 {len(days)} 日を回す (sleep={a.sleep}s)")
    for i, d in enumerate(days, 1):
        r = collect_day(con, d, Path(a.cache), a.force)
        con.commit()
        print(f"[{i}/{len(days)}] {d} -> {r}", flush=True)
        time.sleep(a.sleep)

    n = con.execute("SELECT COUNT(*) FROM paper_comment").fetchone()[0]
    print(f"\npaper_comment 合計 {n} 行")
    for s, c in con.execute("SELECT status, COUNT(*) FROM paper_done GROUP BY status ORDER BY 2 DESC"):
        print(f"  {s}: {c} 日")


if __name__ == "__main__":
    main()
