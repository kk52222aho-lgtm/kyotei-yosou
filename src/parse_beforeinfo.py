"""`beforeinfo_raw` の生文字列を艇ごとの表にほどく(部品交換・調整重量)。

## なんで表にするだけで止めるか

`beforeinfo_raw` は **20260717〜 の 6,455レース**しかない(全253,035レースの **2.5%**)。
`race_z` は NaN が1つ在るレースを丸ごと落とすんで、**いま特徴量に足したら
盤の使えるレースが崩壊する**。せやから

    いまやる  : 生文字列を艇ごとの数にほどいて表に置く(検算つき)
    やらん    : features.py に載せる。**1年貯まってから**

## 測った上で分かっとること(2026-09-26)

- `chosei`(調整重量)は **体重でほぼ決まる**(chosei vs weight r=**−0.5554**)。
  最低重量に合わせる重りやから当然で、盤には `weight` が既に在る。
  体重 ≤46kg で中央2.0kg / ≥52kg でほぼ0。**新情報は残差の分だけ**
- `parts`(部品交換)は盤にも公式B/Kにも無い量。レースの **24.7%** で発生。
  いちばん多いんは**リング**(1,285件)で、キャブ231・ピストン201・ギヤ181・
  シリンダ124・電気121・シャフト49・キャリボ7
- 🚨 `entries.weight_today` は **6,455レース全部 NULL**。書き手がおらん死んだ列
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata

# 「2:ピストン×2 電気 | 5:リング×2」の形。艇番 → 部品の列挙
PARTS_TOK = re.compile(r"([1-6])\s*:\s*([^|]+)")
# 部品名と個数。「リング×2」「電気」
PARTS_ONE = re.compile(r"([^\s×0-9]+)(?:\s*×\s*(\d+))?")
CHOSEI_TOK = re.compile(r"([1-6])\s*:\s*([\d.]+)")

# 部品名の正規化。表記ゆれを寄せる(短い名前が長い名前の一部になるんで長い順に見る)
PART_CANON = [
    ("キャリボ", "carrier"), ("キャブ", "carb"), ("ピストン", "piston"),
    ("リング", "ring"), ("シリンダ", "cylinder"), ("ギヤ", "gear"),
    ("クランク", "crank"), ("シャフト", "shaft"), ("電気", "electric"),
]

DDL = """
CREATE TABLE IF NOT EXISTS beforeinfo_parsed (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER,
  chosei REAL,          -- 調整重量(kg)。体重でほぼ決まる(r=-0.56)
  parts_n INTEGER,      -- 交換した部品の個数(×2 は 2 で数える)
  parts_kinds TEXT,     -- 正規化した部品名をカンマで(ring,carb…)。無交換は空
  parsed_at TEXT,
  PRIMARY KEY (date, jcd, rno, lane)
);
"""


def parse_parts(s: str | None) -> dict[int, tuple[int, list[str]]]:
    """'2:ピストン×2 電気 | 5:リング' → {2:(3,['piston','electric']), 5:(1,['ring'])}

    🚨 個数は**全角数字**(×２)で来る。NFKC で寄せてから数える。
    最初の版は `[^\s×0-9|:]+` で ASCII 数字だけ除いとったんで、
    全角の２が「拾えん部品名」として 1,355件も unknown に入っとった
    (ring の 1,285 より多くて、それで気付いた)。
    """
    out: dict[int, tuple[int, list[str]]] = {}
    if not s or not s.strip():
        return out
    s = unicodedata.normalize("NFKC", s)
    for lane, body in PARTS_TOK.findall(s):
        n = 0
        kinds: list[str] = []
        rest = body
        for jp, en in PART_CANON:
            for m in re.finditer(jp + r"(?:\s*×\s*(\d+))?", rest):
                n += int(m.group(1)) if m.group(1) else 1
                if en not in kinds:
                    kinds.append(en)
            rest = rest.replace(jp, " ")
        # 🚨 拾えん語が残っとったら**黙って捨てん**。unknown で数える
        leftover = [w for w in re.findall(r"[^\s×0-9|:]+", rest) if w.strip()]
        if leftover:
            n += len(leftover)
            kinds.append("unknown")
        out[int(lane)] = (n, kinds)
    return out


def parse_chosei(s: str | None) -> dict[int, float]:
    if not s or not s.strip():
        return {}
    return {int(l): float(v)
            for l, v in CHOSEI_TOK.findall(unicodedata.normalize("NFKC", s))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(DDL)
    import time
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    rows = con.execute(
        "SELECT date, jcd, rno, parts_raw, chosei_raw FROM beforeinfo_raw").fetchall()
    n_race = ins = 0
    unknown_races = 0
    for date, jcd, rno, praw, craw in rows:
        pp = parse_parts(praw)
        cc = parse_chosei(craw)
        if not pp and not cc:
            continue
        n_race += 1
        if any("unknown" in k for _, k in pp.values()):
            unknown_races += 1
        for lane in sorted(set(pp) | set(cc)):
            n, kinds = pp.get(lane, (0, []))
            con.execute(
                "INSERT OR REPLACE INTO beforeinfo_parsed VALUES (?,?,?,?,?,?,?,?)",
                (date, str(jcd).zfill(2), int(rno), lane,
                 cc.get(lane), n, ",".join(kinds), now))
            ins += 1
    con.commit()

    print(f"beforeinfo_raw {len(rows):,}レース → ほどけた {n_race:,}レース / {ins:,}艇")
    print(f"  🚨 拾えん部品名が残ったレース: {unknown_races}"
          + ("(unknown で数えとる)" if unknown_races else ""))
    q = con.execute(
        "SELECT COUNT(*), SUM(chosei IS NOT NULL), SUM(parts_n>0),"
        " SUM(parts_kinds LIKE '%unknown%') FROM beforeinfo_parsed").fetchone()
    print(f"  表: {q[0]:,}艇 / chosei 非空 {q[1]:,} / 部品交換あり {q[2]:,}"
          f" / unknown を含む {q[3]:,}")
    print("  部品の内訳:")
    from collections import Counter
    cnt: Counter = Counter()
    for (k,) in con.execute(
            "SELECT parts_kinds FROM beforeinfo_parsed WHERE parts_kinds<>''"):
        for x in k.split(","):
            cnt[x] += 1
    for k, v in cnt.most_common():
        print(f"    {k:10s} {v:,}")
    print("\n🚨 これは features.py に載せん。覆っとるんが全レースの2.5%で、"
          "載せたら race_z が使えるレースを丸ごと落とす。1年貯まってから。")


if __name__ == "__main__":
    main()
