"""保存済み beforeinfo_raw から **展示ST** を掘り直す(再取得なし・完全オフライン)。

=== 何が壊れとったか ===
`scraper.fetch_beforeinfo` はスタート展示テーブルを **tr ごとに** 読んで
`^([1-6])\\s+(F?\\.?\\d+)` を当てとる。やが今のページは**テーブル全体が1行に潰れる**:

    スタート展示 コース 並び ST 1 .31 2 .14 3 .09 4 .02 5 .01 6 .01

行頭が「スタート展示」なので正規表現が当たらん。結果:
  - `beforeinfo_raw.st_tenji_raw` には 3,455レース分ちゃんと入っとる
  - `entries.tenji_st` は 202607/202608 とも **0行**

**生テキストは録れとって、構造化列に落ちてへん**いう配管の切れ方や。
これが「展開7特徴のうち4本(st_rank/st_z/st_gap/inner_st_edge)が
歴史データで std=0 の定数」の直接の原因になっとる。

=== 直し方 ===
tr 単位やのうて**テキスト全体から (枠, ST) の対を拾う**。6対そろった時だけ採る。
そろわんレース(取得が早すぎて表が空)は **NULL のまま残す**。0 で埋めたら
「まだ出とらん」が「ST 0.00 の絶好のスタート」に化ける。

なお、この表の**並び順が展示の進入隊形**や。K ファイルの実進入コースと違うて
**レース前に見える**量なので、列 tenji_course として一緒に落とす。

出どころは DB の beforeinfo_raw のみ。通信は一切せん。

usage: python -u -m src.backfill_tenji_st
"""
from __future__ import annotations

import re

import pandas as pd

from . import storage

PAIR = re.compile(r"([1-6])\s+([FL]?\.\d+)")

DDL = """
CREATE TABLE IF NOT EXISTS tenji_st (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER,
  tenji_st REAL, tenji_st_f INTEGER, tenji_course INTEGER,
  PRIMARY KEY (date, jcd, rno, lane)
)
"""


def parse(raw: str):
    """(lane, st, flag, course) の6件。そろわんかったら None。"""
    if not raw:
        return None
    m = PAIR.findall(raw)
    if len(m) != 6:
        return None
    lanes = [int(a) for a, _ in m]
    if sorted(lanes) != [1, 2, 3, 4, 5, 6]:
        return None                       # 枠が重複/欠け = 読み違い。捨てる
    out = []
    for course, (a, b) in enumerate(m, start=1):
        flag = 1 if b[0] in "FL" else 0
        try:
            st = float(b[1:] if flag else b)
        except ValueError:
            return None
        if not (0 <= st <= 2.0):
            return None
        out.append((int(a), st, flag, course))
    return out


def main():
    conn = storage.connect()
    conn.execute(DDL)
    d = pd.read_sql_query(
        "SELECT date, jcd, rno, st_tenji_raw FROM beforeinfo_raw", conn)
    ok = miss = 0
    rows = []
    for _, r in d.iterrows():
        p = parse(r["st_tenji_raw"] or "")
        if p is None:
            miss += 1
            continue
        ok += 1
        for lane, st, f, course in p:
            rows.append((r["date"], str(r["jcd"]).zfill(2), int(r["rno"]),
                         lane, st, f, course))
    conn.executemany("INSERT OR REPLACE INTO tenji_st VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()

    print(f"beforeinfo_raw {len(d):,}レース")
    print(f"  6艇そろって読めた : {ok:,} ({ok / len(d):.1%})  = {len(rows):,}行")
    print(f"  読めんかった      : {miss:,}  ← 表が空(取得が早すぎ)。NULLのまま、0埋めせん")

    q = pd.read_sql_query("""
        SELECT COUNT(*) n, COUNT(DISTINCT date) d, MIN(date) d0, MAX(date) d1,
               SUM(tenji_st_f) fl, SUM(tenji_course<>lane) diff, AVG(tenji_st) mst
        FROM tenji_st""", conn)
    print(f"\n=== tenji_st ===\n{q.to_string(index=False)}")
    print("\n  展示の進入隊形 vs 枠番(コース別):")
    print(pd.read_sql_query("""
        SELECT lane, COUNT(*) n, SUM(tenji_course<>lane) diff,
               ROUND(AVG(tenji_st), 3) avg_st
        FROM tenji_st GROUP BY lane ORDER BY lane""", conn).to_string(index=False))

    # 本番の実進入コース(results_st)と突き合わせて、展示がどれだけ当たるか
    j = pd.read_sql_query("""
        SELECT t.tenji_course, r.course
        FROM tenji_st t JOIN results_st r
          ON t.date=r.date AND t.jcd=r.jcd AND t.rno=r.rno AND t.lane=r.lane""", conn)
    if len(j):
        hit = (j["tenji_course"] == j["course"]).mean()
        print(f"\n  展示の進入隊形が本番の進入コースと一致: {hit:.1%}  (n={len(j):,}行)")
        print("  → レース前に見える進入予測としてどれだけ使えるかの目安")
    conn.close()


if __name__ == "__main__":
    main()
