"""オリジナル展示データ (一周/まわり足/直線タイム) の収集。

BOATCAST が全24場の「オリジナル展示データ」を単一ホストの静的TSVで集約している。
公式 boatrace.jp の出走表/直前情報には出ない、各場が独自計測した数値。

エンドポイント (認証なし・素のGETで200・1レース約230-265バイト):
    https://race.boatcast.jp/txt/{jcd}/bc_oriten_{YYYYMMDD}_{jcd}_{RR}.txt   (RR=2桁ゼロ埋め)

形式 (先頭 data=、ヘッダ行にラベル、以降 艇番→選手名→数値):
    data=
    1\t3
    一　周\tまわり足\t直　線
    1\t南　　　佑典\t37.44\t6.26\t7.39
    ...
列構成は場依存 (尼崎は2列 等)。ヘッダのラベルで対応付けし、想定外ラベルは raw に退避。

運営への配慮:
- まず getEncodeList2_Race_{date}.json (全場1リクエスト/日) で開催レースだけ列挙し、
  存在しないレースを叩かない (403の乱射を避ける)。
- リクエスト間に SLEEP_SEC の待機。
- 取得済み (date,jcd,rno) は oriten_done に記録し、再実行では再取得しない。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date as _date, datetime, timedelta

import requests

# 2026-09-08: 素の `import storage` は src/ を cwd にした時しか通らんかった。
# タスク登録して `python -m src.collect_oriten` で呼べるように相対importへ。
from .venues import ALL_JCD, name as venue_name
from . import storage

HOST = "https://race.boatcast.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (kyotei-yosou research; personal use)"}
SLEEP_SEC = 1.5  # マナーとしての待機 (公式より一段丁寧に)
# 403(頻度制限)を食らった時の冷却。実測で約10秒で戻るんで余裕を持たせる
BLOCK_WAIT_SEC = 15.0

_session = requests.Session()
_session.headers.update(HEADERS)

# ヘッダラベル (全角空白を除去して正規化) -> カラム名
# 注意: 場ごとにラベルもスケールも異なる。桐生は「一周」でなく「半周ラップ」を公開し、
# まわり足の値域も大村~6 と 住之江/尼崎~11.5 で桁が違う(計測定義が場依存)。
# したがって生値の場横断比較は不可。特徴量はレース内順位/正規化で使うこと。
LABEL_MAP = {
    "一周": "lap",
    "半周ラップ": "halflap",
    "まわり足": "mawariashi",
    "直線": "straight",
}

ORITEN_SCHEMA = """
CREATE TABLE IF NOT EXISTS oriten (
    date       TEXT NOT NULL,
    jcd        TEXT NOT NULL,
    rno        INTEGER NOT NULL,
    lane       INTEGER NOT NULL,
    lap        REAL,
    halflap    REAL,
    mawariashi REAL,
    straight   REAL,
    raw        TEXT,
    fetched_at TEXT,
    PRIMARY KEY (date, jcd, rno, lane)
);
"""

DONE_SCHEMA = """
CREATE TABLE IF NOT EXISTS oriten_done (
    date   TEXT NOT NULL,
    jcd    TEXT NOT NULL,
    rno    INTEGER NOT NULL,
    status TEXT,          -- 'ok' | 'empty' | 'error'
    n_lane INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (date, jcd, rno)
);
"""


def _init(conn) -> None:
    # ロック競合時は最大60秒待つ(他プロセスのDB利用と共存)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(ORITEN_SCHEMA)
    conn.execute(DONE_SCHEMA)
    # 既存DBに halflap 列が無ければ追加
    cols = {r[1] for r in conn.execute("PRAGMA table_info(oriten)")}
    if "halflap" not in cols:
        conn.execute("ALTER TABLE oriten ADD COLUMN halflap REAL")
    conn.commit()


def _get_text(url: str, retries: int = 4) -> tuple[int, str]:
    """(status_code, text) を返す。最終試行の status を返す。"""
    status = 0
    for i in range(retries):
        try:
            r = _session.get(url, timeout=20)
            status = r.status_code
            if status == 200:
                return 200, r.content.decode("utf-8", "replace")
            if status == 403:
                # 🚨 2026-09-24: ここで「403 = 不在。リトライ不要」と決め打っとった。
                #    実測では **403 は頻度制限**で、不在やない。
                #    1.5秒間隔で叩くと **14発で引っかかって、約10秒で戻る**
                #    (25発の実測: 200×18 / 403×7、15発目〜21発目が403)。
                #    empty と記帳された 9/17 以降のレースを叩き直したら
                #    **5/5 が 200 で中身を返した**。8日ぶん静かに落としとった。
                #    冷却を待って同じレースをやり直す。それでも駄目なら blocked。
                time.sleep(BLOCK_WAIT_SEC * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(SLEEP_SEC * (i + 1))
    return status, ""


def enumerate_day(date: str) -> dict[str, int]:
    """getEncodeList2 で当日開催の {jcd: 最大レース番号} を返す (全場1リクエスト)。"""
    url = f"{HOST}/api_txt/getEncodeList2_Race_{date}.json"
    status, text = _get_text(url)
    out: dict[str, int] = {}
    if status != 200 or not text:
        return out
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return out
    for r in d.get("return_info", []):
        jcd = str(r.get("RaceStudiumNo", "")).zfill(2)
        hi = r.get("Holding_info", [])
        if hi:
            n = len(hi[0].get("Status_Info", []))
            if n:
                out[jcd] = n
    return out


def parse_oriten(text: str) -> list[dict]:
    """oriten TSV をパースして [{lane, lap, mawariashi, straight, raw}] を返す。"""
    lines = [ln for ln in text.split("\n")]
    # data= を除去
    lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("data=")]
    # ヘッダ行を探す (一/まわり/直 を含む)
    header_idx = None
    labels: list[str] = []
    for i, ln in enumerate(lines):
        cells = ln.split("\t")
        norm = [c.replace("　", "").replace(" ", "") for c in cells]
        if any(k in "".join(norm) for k in ("一周", "まわり足", "直線")):
            header_idx = i
            labels = norm
            break
    if header_idx is None:
        return []
    rows = []
    for ln in lines[header_idx + 1:]:
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        try:
            lane = int(parts[0])
        except ValueError:
            continue
        if lane < 1 or lane > 6:
            continue
        values = parts[2:]  # 選手名の後が数値列
        rec = {"lane": lane, "lap": None, "halflap": None,
               "mawariashi": None, "straight": None, "raw": ln}
        for label, val in zip(labels, values):
            col = LABEL_MAP.get(label)
            v = None
            try:
                v = float(val)
            except (ValueError, TypeError):
                v = None
            if col:
                rec[col] = v
        rows.append(rec)
    return rows


def fetch_race(conn, date: str, jcd: str, rno: int) -> str:
    """1レース分を取得してDBへ。status ('ok'|'empty'|'error') を返す。"""
    rr = str(rno).zfill(2)
    url = f"{HOST}/txt/{jcd}/bc_oriten_{date}_{jcd}_{rr}.txt"
    status, text = _get_text(url)
    now = datetime.now().isoformat(timespec="seconds")
    if status == 200 and text:
        rows = parse_oriten(text)
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO oriten "
                "(date,jcd,rno,lane,lap,halflap,mawariashi,straight,raw,fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(date, jcd, rno, r["lane"], r["lap"], r["halflap"],
                  r["mawariashi"], r["straight"], r["raw"], now) for r in rows],
            )
            _mark_done(conn, date, jcd, rno, "ok", len(rows), now)
            return "ok"
        _mark_done(conn, date, jcd, rno, "empty", 0, now)
        return "empty"
    if status == 200:
        # 200 やのに中身が無い = ほんまに不在。ここだけが empty
        _mark_done(conn, date, jcd, rno, "empty", 0, now)
        return "empty"
    # 🚨 403/その他は「不在」やない。**blocked として残して次回やり直す**
    _mark_done(conn, date, jcd, rno, "blocked" if status == 403 else "error", 0, now)
    return "blocked" if status == 403 else "error"


def _mark_done(conn, date, jcd, rno, status, n_lane, now) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO oriten_done "
        "(date,jcd,rno,status,n_lane,fetched_at) VALUES (?,?,?,?,?,?)",
        (date, jcd, rno, status, n_lane, now),
    )


def _already_done(conn, date, jcd, rno) -> bool:
    """済みは ok と empty(=200やのに中身が無い)だけ。

    blocked/error は**済みやない**。弾かれただけのもんを「不在」に格上げしたら、
    そのレースは永久に取れんくなる。→ [[insight_zero_is_a_measurement]]
    """
    r = conn.execute(
        "SELECT 1 FROM oriten_done WHERE date=? AND jcd=? AND rno=? "
        "AND status IN ('ok','empty')",
        (date, jcd, rno)).fetchone()
    return r is not None


def collect_day(conn, date: str, jcds: list[str] | None = None,
                verbose: bool = True) -> dict[str, int]:
    """1日分を収集。{'ok','empty','error','skip'} の件数を返す。"""
    # やることが無い日は**通信せず**飛ばす。
    # 🚨 403 の退きを長くしたら、列挙の1発にも 15+30+45+60 秒かかるようになって、
    #    済みだけの日を舐めるのに何時間もかかった。まず台帳で用事の有無を見る。
    have = conn.execute("SELECT COUNT(*) FROM oriten_done WHERE date=?", (date,)).fetchone()[0]
    todo = conn.execute(
        "SELECT COUNT(*) FROM oriten_done WHERE date=? AND status NOT IN ('ok','empty')",
        (date,)).fetchone()[0]
    if have and not todo:
        return {"ok": 0, "empty": 0, "error": 0, "skip": have, "blocked": 0}
    plan = enumerate_day(date)
    time.sleep(SLEEP_SEC)
    if jcds:
        plan = {k: v for k, v in plan.items() if k in jcds}
    counts = {"ok": 0, "empty": 0, "error": 0, "skip": 0, "blocked": 0}
    if not plan:
        if verbose:
            print(f"{date}: 開催なし (or 列挙失敗)")
        return counts
    for jcd in sorted(plan):
        nmax = plan[jcd]
        for rno in range(1, nmax + 1):
            if _already_done(conn, date, jcd, rno):
                counts["skip"] += 1
                continue
            st = fetch_race(conn, date, jcd, rno)
            counts[st] += 1
            time.sleep(SLEEP_SEC)
        conn.commit()
        if verbose:
            print(f"  {date} {jcd}{venue_name(jcd)}: "
                  f"ok={counts['ok']} empty={counts['empty']} "
                  f"error={counts['error']} skip={counts['skip']}")
    return counts


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="オリジナル展示データ収集 (BOATCAST)")
    ap.add_argument("--start", required=True, help="開始日 YYYYMMDD")
    ap.add_argument("--end", help="終了日 YYYYMMDD (省略時は --start と同じ)")
    ap.add_argument("--jcd", nargs="*", help="場コード指定 (省略時は全場)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    end = args.end or args.start
    jcds = [j.zfill(2) for j in args.jcd] if args.jcd else None

    conn = storage.connect()
    _init(conn)
    total = {"ok": 0, "empty": 0, "error": 0, "skip": 0}
    for date in daterange(args.start, end):
        c = collect_day(conn, date, jcds, verbose=not args.quiet)
        for k in total:
            total[k] += c[k]
    conn.close()
    print(f"\n完了: ok={total['ok']} empty={total['empty']} "
          f"error={total['error']} skip={total['skip']}")


if __name__ == "__main__":
    main()
