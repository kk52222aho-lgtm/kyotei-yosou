"""宮島の紙を毎日取って、**締切前に**凍結モデルの予測を記帳する。

holdout は2回見た。**残っとるのは前向きだけ**や(`docs/prereg_paper_ngram.md`)。

やること(順番に意味がある):

1. `src.collect_paper`      … その日の紙からコメントを退避(コーパスを止めん)
2. `src.collect_paper_marks`… 記者の予想印も退避
3. **凍結モデルで採点して `paper_forward` に記帳**

## 記帳の規則

- **着順も払戻も一切見ずに保存する。**紙には結果が刷られとるが、
  `collect_paper` は列として持たん作りにしてある
- `recorded_at` と **その日の紙に刷られた締切時刻**を必ず残す。
  判定では **`deadline` > `recorded_at` のレースだけ**を数える。
  「締切前やった」を**レース単位で検算できる**ようにしとく
- 凍結モデルの SHA も毎行に残す。**すり替えを検算できるように**
- 既に記帳した (date,rno,lane) は**上書きせん**。最初の記帳が証人や

## 紙はいつ出るか(実測)

2026-09-22 の紙は **11:47 の時点で 16:36 締切の R12 のコメントまで全部在った**。
= 最終レースの5時間前。朝に1回回せば全レース間に合う。
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np

from .collect_paper import JCD

MODEL_PATH = "data/paper_ngram.joblib"

DDL = """
CREATE TABLE IF NOT EXISTS paper_forward (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER, reg TEXT,
  ngram_score REAL, mark TEXT, comment TEXT,
  deadline TEXT, model_sha TEXT, recorded_at TEXT,
  PRIMARY KEY (date, jcd, rno, lane)
);
"""


def _deadlines(date: str) -> dict[int, str]:
    """その日の紙に刷られた締切時刻を {レース番号: "HH:MM"} で返す。

    判定で「ほんまに締切前に記帳したか」をレース単位で確かめるために要る。
    紙の左端(x≈21)に1レース1個ずつ縦に並んどる。取れんかったら空で返す
    (**空を「締切前やった」に化けさせん**ため、判定側は空の行を数えん)。
    """
    import io as _io
    import re as _re
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer
    p = Path("data/paper/17") / f"{date}.pdf"
    if not p.exists():
        return {}
    out: dict[int, str] = {}
    rno = 0
    try:
        for page in extract_pages(_io.BytesIO(p.read_bytes()),
                                  laparams=LAParams(detect_vertical=True, line_margin=0.3)):
            hits = [(e.bbox[3], e.get_text().strip()) for e in page
                    if isinstance(e, LTTextContainer) and e.bbox[0] < 40
                    and _re.fullmatch(r"\d{1,2}:\d{2}", e.get_text().strip())]
            for _, t in sorted(hits, key=lambda z: -z[0]):
                rno += 1
                out[rno] = t
    except Exception:
        return {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--date", default=None, help="YYYYMMDD(既定は今日)")
    ap.add_argument("--model", default=MODEL_PATH)
    a = ap.parse_args()
    date = a.date or time.strftime("%Y%m%d")

    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(DDL)
    con.commit()

    n_ent = con.execute(
        "SELECT COUNT(*) FROM entries WHERE date=? AND jcd=?", (date, JCD)).fetchone()[0]
    if not n_ent:
        print(f"{date}: 宮島は開催なし(entries に行が無い)。何もせん")
        return

    # 1-2. コーパスを止めん。既にやってあったら中で飛ばされる
    import runpy
    import sys
    for mod, extra in (("src.collect_paper", ["--from", date, "--to", date, "--sleep", "1"]),
                       ("src.collect_paper_marks", ["--from", date, "--to", date])):
        argv = sys.argv
        sys.argv = [mod] + extra
        try:
            runpy.run_module(mod, run_name="__main__")
        except SystemExit:
            pass
        finally:
            sys.argv = argv

    # 3. 凍結モデルで採点して記帳
    mp = Path(a.model)
    if not mp.exists():
        print(f"🚨 凍結モデルが無い: {mp}。`python -m src.freeze_paper_ngram` を先に")
        return
    sha = hashlib.sha256(mp.read_bytes()).hexdigest()
    bundle = joblib.load(mp)
    vec, model = bundle["vec"], bundle["model"]

    deadlines = _deadlines(date)
    rows = con.execute(
        "SELECT c.rno, c.lane, c.reg, c.comment, COALESCE(m.mark,'') "
        "FROM paper_comment c LEFT JOIN paper_mark m "
        "  ON c.date=m.date AND c.jcd=m.jcd AND c.rno=m.rno AND c.lane=m.lane "
        "WHERE c.date=? AND c.jcd=?", (date, JCD)).fetchall()
    if not rows:
        print(f"{date}: コメントが取れとらん。記帳せん")
        return
    texts = [unicodedata.normalize("NFKC", r[3] or "") for r in rows]
    score = model.predict(vec.transform(texts))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    ins = 0
    for (rno, lane, reg, comment, mark), s in zip(rows, score):
        # 🚨 上書きせん。最初の記帳が証人や
        cur = con.execute(
            "INSERT OR IGNORE INTO paper_forward VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (date, JCD, int(rno), int(lane), str(reg), float(s), mark, comment,
             deadlines.get(int(rno), ""), sha, now))
        ins += cur.rowcount
    con.commit()
    tot = con.execute("SELECT COUNT(*) FROM paper_forward").fetchone()[0]
    nrace = con.execute(
        "SELECT COUNT(DISTINCT rno) FROM paper_forward WHERE date=?", (date,)).fetchone()[0]
    print(f"{date}: 記帳 {ins} 行(新規)/ その日 {nrace}レース / 累計 {tot:,} 行")
    print(f"  model_sha={sha[:16]}… recorded_at={now}")
    print(f"  ngram_score: 中央 {np.median(score):+.5f} / 幅 "
          f"[{score.min():+.5f}, {score.max():+.5f}]")


if __name__ == "__main__":
    main()
