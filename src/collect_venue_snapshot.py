"""場公式サイトの「全選手コメント」「記者展望」を毎日そのまま退避する。

上書き型や。**遡及はゼロ**(`targetday` を渡してもサーバが黙って捨てて今日を返す。
津/徳山/浜名湖とも4日付で md5 が1個やった)。1日待てば1日ぶん永久に消える。
→ [[insight_perishability_first]]

## 退避と抽出は分ける

ここは**生HTMLを gzip で置くだけ**。パースはせん。
チラシで学んだ形(退避は全部落とす / 抽出は後から追いかける)。
今日パースでけへん面でも、明日パーサを直せば過去ぶんに当たれる。

## 🚨 的は推測せん。実在を数えてから載せる

`page=` に知らん値を渡すと**黙って索引ページに落ちる**。
若松で `index_racers_comment` を叩くと 200 が返って2,295字の本文が出るが、
そんなページは無い。**「200が返った」も「本文があった」も存在証明にならん。**

2026-09-23 に24場の トップページから `page=` の値を全部数えて確定した:

    index_racers_comment … 芦屋・唐津の **2場だけ**
    index_tenbo          … 10場
    (9場はこのCMSやない = 別の作りで、まだ調べとらん。「無い」やのうて「未調査」)

数え直すたび数が減っとる(13場 → 12場 → 10場)。推測で広げたら毎回こうなる。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (compatible; kyotei-yosou research collector)"

# 2026-09-23 に実在を数えて確定した的。増やすときは数え直してから
TARGETS: list[tuple[str, str, str]] = [
    # (jcd, ページ名, ドメイン)
    ("21", "index_racers_comment", "www.boatrace-ashiya.com"),
    ("23", "index_racers_comment", "www.boatrace-karatsu.jp"),
    ("05", "index_tenbo", "www.boatrace-tamagawa.com"),
    ("06", "index_tenbo", "www.boatrace-hamanako.jp"),
    ("08", "index_tenbo", "www.boatrace-tokoname.jp"),
    ("10", "index_tenbo", "www.boatrace-mikuni.jp"),
    ("11", "index_tenbo", "www.boatrace-biwako.jp"),
    ("13", "index_tenbo", "www.boatrace-amagasaki.jp"),
    ("14", "index_tenbo", "www.n14.jp"),
    ("18", "index_tenbo", "www.boatrace-tokuyama.jp"),
    ("19", "index_tenbo", "www.boatrace-shimonoseki.jp"),
    ("23", "index_tenbo", "www.boatrace-karatsu.jp"),
]

URL = "https://{dom}/modules/raceinfo/?page={page}"

# 中身が入っとるかの安い目印。**「開催が無いから空」と「取り損ねて空」を分けるため**に
# 0 も必ず記録する。→ [[insight_zero_is_a_measurement]]
BODY_WORDS = re.compile(r"ペラ|プロペラ|回転|伸び|エンジン|機力|出足|行き足|乗り")
JP_RUN = re.compile(r"[ぁ-んァ-ヶ一-龥、。…・]{12,}")

DDL = """
CREATE TABLE IF NOT EXISTS venue_snap (
  date TEXT, jcd TEXT, page TEXT,
  http INTEGER, n_bytes INTEGER, sha256 TEXT,
  n_body_hits INTEGER, path TEXT, fetched_at TEXT,
  PRIMARY KEY (date, jcd, page, fetched_at)
);
"""


def body_hits(html: str) -> int:
    t = re.sub(r"(?is)<script.*?</script>", "", html)
    t = re.sub(r"<[^>]+>", " ", t)
    return sum(1 for s in JP_RUN.findall(t) if BODY_WORDS.search(s))


def run(db: str, out: str, sleep: float) -> None:
    con = sqlite3.connect(db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(DDL)
    date = time.strftime("%Y%m%d")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    d = Path(out) / date
    d.mkdir(parents=True, exist_ok=True)
    got = hit = 0
    for jcd, page, dom in TARGETS:
        url = URL.format(dom=dom, page=page)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                code, raw = r.status, r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  {jcd} {page}: 失敗 {e}")
            con.execute("INSERT OR REPLACE INTO venue_snap VALUES (?,?,?,?,?,?,?,?,?)",
                        (date, jcd, page, 0, 0, "", -1, "", now))
            continue
        html = raw.decode("utf-8", "replace")
        n = body_hits(html)
        p = d / f"{jcd}_{page}.html.gz"
        p.write_bytes(gzip.compress(raw))
        con.execute("INSERT OR REPLACE INTO venue_snap VALUES (?,?,?,?,?,?,?,?,?)",
                    (date, jcd, page, code, len(raw),
                     hashlib.sha256(raw).hexdigest(), n, str(p), now))
        got += 1
        hit += n > 0
        print(f"  {jcd} {page:22s} http={code} {len(raw):7d}B 本文らしき塊={n}")
        time.sleep(sleep)
    con.commit()
    print(f"\n{date}: {got}/{len(TARGETS)} 退避、うち中身あり {hit}")
    print("(中身0は『開催が無い』かもしれん。0も台帳に残しとる)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--out", default="data/venue_snap")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    run(a.db, a.out, a.sleep)


if __name__ == "__main__":
    main()
