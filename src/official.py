"""公式オープンデータ (boatrace.jp ダウンロード) からの一括収集。

  番組表 B ファイル: https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh
  競走成績 K ファイル: https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh

1ファイルで全24場・全レースを含む（Shift_JIS 固定幅テキストを LZH 圧縮）。
スクレイピングと違い 1 日 2 ファイルの DL で済み、過去分も大量に取得できる。

B = 番組（事前情報: 勝率・2率・モーター/ボート2率 など）
K = 結果（着順）＋ 展示タイム ＋ 天候/風/波 ＋ 払戻金

注意: K の「進入コース・実ST・レースタイム」はレース後確定のためモデル特徴量には使わない
（展示タイム・天候系のみ事前情報として利用）。

例:
  python -m src.official --start 20260401 --end 20260629
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time

import lhafile
import requests

from . import storage

RAW_DIR = os.path.join(storage.DATA_DIR, "raw")
HEADERS = {"User-Agent": "Mozilla/5.0 (kyotei-yosou research; personal use)"}
SLEEP_SEC = 0.5

# B 番組: 艇番 登番 名前 年齢 支部 体重 級別  全国勝率 全国2率 当地勝率 当地2率 ﾓｰﾀｰNO ﾓｰﾀｰ2率 ﾎﾞｰﾄNO ﾎﾞｰﾄ2率 ...
B_RACER = re.compile(
    r"^([1-6]) (\d{4})(\D+?)(\d{2})(\D+?)(\d{2})(A1|A2|B1|B2)\s+(.+)$")
# K 結果ヘッダ: "  1R ... H1800m  晴  風 南西  2m  波  1cm"
K_HEADER = re.compile(
    r"^\s*(\d{1,2})R\s+.*?H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+([\d.]+)m\s+波\s+([\d.]+)cm")
# K 着順行: "  01  3 3964 宮 地 博 士 30  22  6.92   3   0.12   1.51.7"
K_RACER = re.compile(
    r"^\s*(\d{2})\s+([1-6])\s+(\d{4})\s+\S.+?\S\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([1-6])\s+([F\d.]+)")
VENUE_MARK = re.compile(r"^(\d{2})(?:BBGN|KBGN)")


def _download(kind: str, date: str) -> bytes | None:
    """kind='B' or 'K'。lzh をDL（data/raw にキャッシュ）して中身の bytes を返す。"""
    yymmdd = date[2:]
    fname = f"{kind.lower()}{yymmdd}.lzh"
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        url = f"https://www1.mbrace.or.jp/od2/{kind}/{date[:6]}/{fname}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            return None
        if r.status_code != 200 or not r.content:
            return None
        with open(path, "wb") as f:
            f.write(r.content)
        time.sleep(SLEEP_SEC)
    try:
        lf = lhafile.Lhafile(path)
        return lf.read(lf.namelist()[0])
    except Exception:
        return None


def parse_b(data: bytes) -> dict:
    """B番組をパース。返り値 {(jcd,rno): [racer dict, ...]}"""
    out: dict[tuple[str, int], list[dict]] = {}
    jcd = None
    rno = None
    for ln in data.decode("shift_jis", "replace").splitlines():
        mv = VENUE_MARK.match(ln)
        if mv:
            jcd = mv.group(1)
            continue
        mr = re.match(r"^\s*(\d{1,2})Ｒ", ln)
        if mr:
            rno = int(mr.group(1))
            continue
        m = B_RACER.match(ln)
        if m and jcd and rno:
            nums = re.findall(r"-?\d+\.?\d*", m.group(8))
            n = [float(x) for x in nums]
            rec = {
                "lane": int(m.group(1)),
                "reg": m.group(2),
                "name": m.group(3).replace("　", " ").strip(),
                "age": float(m.group(4)),
                "weight": float(m.group(6)),
                "racer_class": m.group(7),
                "nat_win": n[0] if len(n) > 0 else None,
                "nat_2rate": n[1] if len(n) > 1 else None,
                "loc_win": n[2] if len(n) > 2 else None,
                "loc_2rate": n[3] if len(n) > 3 else None,
                "motor_2rate": n[5] if len(n) > 5 else None,
                "boat_2rate": n[7] if len(n) > 7 else None,
            }
            out.setdefault((jcd, rno), []).append(rec)
    return out


# 払戻金テーブル行: "1R  1-3-6   2380   1-3-6   880   1-3   340   1-3   240"
PAYOUT_ROW = re.compile(
    r"^\s*(\d{1,2})R\s+(\d-\d-\d)\s+(\d+)\s+(\d-\d-\d)\s+(\d+)\s+(\d-\d)\s+(\d+)\s+(\d-\d)\s+(\d+)")
TANSHO_ROW = re.compile(r"単勝\s+([1-6])\s+(\d+)")


def parse_payouts(data: bytes) -> dict:
    """K の払戻金をパース。返り値 {(jcd,rno): {3連単/3連複/2連単/2連複/単勝}}

    払戻は 100 円賭けあたりの金額。
    """
    out: dict[tuple[str, int], dict] = {}
    jcd = None
    rno = None
    for ln in data.decode("shift_jis", "replace").splitlines():
        mv = VENUE_MARK.match(ln)
        if mv:
            jcd = mv.group(1)
            continue
        mp = PAYOUT_ROW.match(ln)
        if mp and jcd:
            r = int(mp.group(1))
            out[(jcd, r)] = {
                "trifecta_combo": mp.group(2), "trifecta_yen": int(mp.group(3)),
                "trio_combo": mp.group(4), "trio_yen": int(mp.group(5)),
                "exacta_combo": mp.group(6), "exacta_yen": int(mp.group(7)),
                "quinella_combo": mp.group(8), "quinella_yen": int(mp.group(9)),
                "tansho_lane": None, "tansho_yen": None,
            }
            continue
        mh = K_HEADER.match(ln)
        if mh and jcd:
            rno = int(mh.group(1))
            continue
        mt = TANSHO_ROW.search(ln)
        if mt and jcd and rno and (jcd, rno) in out:
            out[(jcd, rno)]["tansho_lane"] = int(mt.group(1))
            out[(jcd, rno)]["tansho_yen"] = int(mt.group(2))
    return out


def parse_k(data: bytes) -> dict:
    """K結果をパース。返り値 {(jcd,rno): {'finish':{lane:着}, 'tenji':{lane:展示}, 'cond':{...}}}"""
    out: dict[tuple[str, int], dict] = {}
    jcd = None
    rno = None
    cur = None
    for ln in data.decode("shift_jis", "replace").splitlines():
        mv = VENUE_MARK.match(ln)
        if mv:
            jcd = mv.group(1)
            continue
        mh = K_HEADER.match(ln)
        if mh and jcd:
            rno = int(mh.group(1))
            cur = {"finish": {}, "tenji": {}, "cond": {
                "wind_speed": float(mh.group(5)),
                "wave_height": float(mh.group(6)),
            }}
            out[(jcd, rno)] = cur
            continue
        mf = K_RACER.match(ln)
        if mf and cur is not None:
            lane = int(mf.group(2))
            cur["finish"][lane] = int(mf.group(1))
            cur["tenji"][lane] = float(mf.group(6))
    return out


def collect_date(conn, date: str) -> int:
    bdata = _download("B", date)
    kdata = _download("K", date)
    if not bdata or not kdata:
        return 0
    programs = parse_b(bdata)
    results = parse_k(kdata)
    payouts = parse_payouts(kdata)
    saved = 0
    for (jcd, rno), entries in programs.items():
        if len(entries) < 6:
            continue
        res = results.get((jcd, rno))
        finish = res["finish"] if res else None
        before = None
        if res:
            per_lane = {ln: {"tenji_time": t} for ln, t in res["tenji"].items()}
            before = (per_lane, res["cond"])
        storage.save_race(conn, date, jcd, rno, entries, finish, before)
        if (jcd, rno) in payouts:
            storage.save_payouts(conn, date, jcd, rno, payouts[(jcd, rno)])
        saved += 1
    conn.commit()
    return saved


def backfill_payouts(conn) -> int:
    """data/raw にキャッシュ済みの K ファイルから払戻金だけを再パースして保存（再DLなし）。"""
    import glob
    n = 0
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "k*.lzh"))):
        yymmdd = os.path.basename(path)[1:7]
        date = "20" + yymmdd
        try:
            lf = lhafile.Lhafile(path)
            data = lf.read(lf.namelist()[0])
        except Exception:
            continue
        for (jcd, rno), p in parse_payouts(data).items():
            storage.save_payouts(conn, date, jcd, rno, p)
            n += 1
    conn.commit()
    return n


def daterange(start: str, end: str):
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser(description="公式オープンデータ(B/K)から一括収集")
    ap.add_argument("--start", required=True, help="開始日 YYYYMMDD")
    ap.add_argument("--end", required=True, help="終了日 YYYYMMDD")
    args = ap.parse_args()

    conn = storage.connect()
    total = 0
    for date in daterange(args.start, args.end):
        n = collect_date(conn, date)
        total += n
        print(f"  {date}: {n} レース")
    conn.close()
    print(f"\n完了: {total} レースを保存")


if __name__ == "__main__":
    main()
