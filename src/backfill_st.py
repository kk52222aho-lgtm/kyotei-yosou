"""K ファイルから **実ST と 進入コース** を掘り出して盤に載せる(再DLなし・完全オフライン)。

=== なぜ ===
`official.parse_k` は K_RACER の正規表現で **group(7)=進入コース / group(8)=実ST を
既に捕まえとる**のに、着順と展示タイムだけ取って捨てとった。docstring の理由は
「レース後確定やからモデル特徴量には使わん」。**そのレース**の分は正しい。

やが **過去のレースの実ST は次のレースの事前情報や。** そして競艇で一番見られとる量が
これや。出走表に載る「平均ST」はコースをまたいだ合算で、しかも盤には
**1行も入っとらん**(entries.avg_st の非欠測 0.2%、しかも2026だけ)。
つまり走査の格子に一番効きそうな量が、そもそも載ってへんかった。

進入コースも同じ。艇番(lane)は入っとるが、**前づけで実際に何コースから出たか**は
別の量で、選手ごとに癖がある。これも一度も盤に載っとらん。

出どころは data/raw/k*.lzh(1,670日分、既にディスク上)。通信は一切せん。

=== 入るもん ===
  results_st(date, jcd, rno, lane, course, st, st_f)
    course : 実際の進入コース(1-6)。lane と違うことがある(前づけ)
    st     : スタートタイミング(秒)。小さいほど速い。欠測は NULL
    st_f   : フライング/出遅れの印(F または L が付いとったら1)

usage: python -u -m src.backfill_st
"""
from __future__ import annotations

import glob
import os
import re

import lhafile

from . import storage
from .official import K_HEADER, VENUE_MARK

RAW_DIR = os.path.join(storage.DATA_DIR, "raw")

# 着 艇 登番 名前 年齢 体重 展示 進入 ST レースタイム
# ST は "0.12" / "F.01"(フライング) / " . "(欠測=出走せず等) を取り得る
K_ROW = re.compile(
    r"^\s*(\d{2}|F|L|K\d|S\d)\s+([1-6])\s+(\d{4})\s+\S.+?\S\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)\s+([1-6])\s+([FL]?[\d.]+)")

DDL = """
CREATE TABLE IF NOT EXISTS results_st (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER,
  course INTEGER, st REAL, st_f INTEGER,
  PRIMARY KEY (date, jcd, rno, lane)
)
"""


def parse_st(data: bytes):
    """[(jcd, rno, lane, course, st, st_f), ...]"""
    rows = []
    jcd = rno = None
    for ln in data.decode("shift_jis", "replace").splitlines():
        mv = VENUE_MARK.match(ln)
        if mv:
            jcd = mv.group(1)
            continue
        mh = K_HEADER.match(ln)
        if mh and jcd:
            rno = int(mh.group(1))
            continue
        m = K_ROW.match(ln)
        if m and jcd and rno:
            raw = m.group(8)
            flag = 1 if raw[0] in "FL" else 0
            body = raw[1:] if flag else raw
            try:
                st = float(body)
            except ValueError:
                st = None
            if st is not None and not (0 <= st <= 2.0):
                st = None
            rows.append((jcd, rno, int(m.group(2)), int(m.group(7)), st, flag))
    return rows


def main():
    conn = storage.connect()
    conn.execute(DDL)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "k*.lzh")))
    print(f"K ファイル {len(files):,} 本(通信なし)")
    total = bad = 0
    for i, path in enumerate(files, 1):
        date = "20" + os.path.basename(path)[1:7]
        try:
            lf = lhafile.Lhafile(path)
            data = lf.read(lf.namelist()[0])
        except Exception as e:
            bad += 1
            print(f"  [壊れ] {path}: {e}")
            continue
        rows = parse_st(data)
        conn.executemany(
            "INSERT OR REPLACE INTO results_st VALUES (?,?,?,?,?,?,?)",
            [(date, j, r, ln, c, s, f) for (j, r, ln, c, s, f) in rows])
        total += len(rows)
        if i % 200 == 0:
            conn.commit()
            print(f"  {i}/{len(files)}  {date}  累計 {total:,}行", flush=True)
    conn.commit()

    cur = conn.execute("""
        SELECT COUNT(*), SUM(st IS NOT NULL), SUM(st_f), SUM(course<>lane),
               MIN(date), MAX(date), COUNT(DISTINCT date)
        FROM results_st""")
    n, nst, nf, ndiff, d0, d1, nd = cur.fetchone()
    print(f"\n=== results_st ===")
    print(f"  {n:,}行 / {nd:,}日 ({d0}-{d1})  壊れファイル {bad}")
    print(f"  ST 非欠測      : {nst:,} ({nst / n:.1%})")
    print(f"  F/L 印         : {nf:,} ({nf / n:.2%})")
    print(f"  進入 != 艇番   : {ndiff:,} ({ndiff / n:.1%})  ← 前づけ。lane では見えん量")
    print("\n  ST の分布(コース別・平均):")
    for row in conn.execute("""SELECT course, COUNT(*), AVG(st) FROM results_st
                               WHERE st IS NOT NULL GROUP BY course ORDER BY course"""):
        print(f"    {row[0]}コース {row[1]:>9,}本  平均ST {row[2]:.3f}")
    conn.close()


if __name__ == "__main__":
    main()
