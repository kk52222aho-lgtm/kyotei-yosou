"""odds_timeseries を日付パーティションでOneDriveへ退避＋完全性(有効N)を吐く。

背景: 軌跡データ(odds_timeseries)はローカルkyotei.db(gitignore)のみ＝無バックアップが
プロジェクト最大の単一障害点。前向き検証は~6週の待ちで、その間にPC/DBが飛べば全消滅。
∴日次で odds_YYYYMMDD.csv をOneDrive(オフサイト・非公開)へ追記型export。公開リポには一切載せん。

設計(既存議論の凍結):
- 日付パーティション追記型: 過去日のファイルは書けたら二度と触らん(OneDriveの削除/破損ミラーに強い)。
  当日ファイルだけ成長するので毎回上書き。
- 検証: 各ファイルが存在し行数>0 を確認してからでないと成功扱いにせん。
- 心拍(export側): 検証通過後だけ HC_ODDS_EXPORT へping(未設定なら省略)。ループ心拍とは別check
  ＝「保存されてる」の独立した沈黙検知。exportがコケてもループpingは貫通するので分離が要る。
- 完全性: 前向き検証の"有効N"(=FLBかつ本命艇が窓の両端mins=10&1を持つレース数)を報告。
  ※凍結条項によりROIは一切見ない。ここで出すのは母集団サイズの進捗カウントだけ。

使い方: python -m src.export_odds       (ループから定期呼び出し or 手動)
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

from . import storage, heartbeat

BACKUP_DIR = os.path.join(
    os.environ.get("OneDrive", os.path.expanduser("~/OneDrive")),
    "kyotei_backup", "odds_partitions")

HC_EXPORT_ENV = "HC_ODDS_EXPORT"   # export後ping(検証通過後のみ)

# odds_timeseries の列(SELECT * の順に合わせて動的取得するので固定はしない)


def _ping(_unused=None):
    """🚨 2026-09-02: 未設定を黙ってskipするのをやめた。heartbeat.ping が吠える。"""
    heartbeat.ping(heartbeat.EXPORT)


def _effective_n(conn) -> tuple[int, float, int]:
    """有効N(FLB×本命艇が窓両端を持つ), 両端揃い率, 生FLB数 を返す。ROIは触らない。"""
    rows = conn.execute(
        "SELECT date,jcd,rno,honmei,combo,mins_to_deadline FROM odds_timeseries "
        "WHERE bet_type='tansho'").fetchall()
    # レース→本命lane, 本命艇が持つminsの集合
    honmei = {}
    hon_mins = {}
    for date, jcd, rno, hon, combo, mins in rows:
        key = (date, jcd, rno)
        if hon is not None:
            honmei[key] = hon
        if hon is not None and str(combo) == str(hon):
            hon_mins.setdefault(key, set()).add(mins)
    flb = [k for k, h in honmei.items() if h != 1]          # 本命≠1号
    eff = [k for k in flb if {10, 1} <= hon_mins.get(k, set())]  # 両端(10&1)揃い
    rate = len(eff) / len(flb) if flb else 0.0
    return len(eff), rate, len(flb)


def export(verbose: bool = True) -> dict:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    conn = sqlite3.connect(storage.DB_PATH, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(odds_timeseries)")]
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM odds_timeseries ORDER BY date")]
    today = dt.date.today()
    today_str = today.strftime("%Y%m%d")
    yesterday_str = (today - dt.timedelta(days=1)).strftime("%Y%m%d")

    written, verified, skipped = [], [], []
    for d in dates:
        path = os.path.join(BACKUP_DIR, f"odds_{d}.csv")
        # 昨日より前で既にファイルあれば不変(写し確定=OneDriveの削除/破損ミラーに強い)。
        # 今日=成長中で毎回上書き。昨日=日付替わり後の最初のexportで一度だけ書き直し尻尾を確定
        # (毎時exportの最終回~22時以降〜締切後の行を取りこぼさんため)。DBが真実・CSVは写し。
        if d < yesterday_str and os.path.exists(path):
            skipped.append(d)
            continue
        recs = conn.execute(
            "SELECT * FROM odds_timeseries WHERE date=? ORDER BY captured_at,jcd,rno,combo",
            (d,)).fetchall()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(recs)
        os.replace(tmp, path)          # 原子的置換(途中破損を残さん)
        written.append(d)
        # 検証: 存在＋行数>0
        if os.path.exists(path) and sum(1 for _ in open(path, encoding="utf-8")) > 1:
            verified.append(d)

    eff_n, rate, raw_flb = _effective_n(conn)
    conn.close()

    all_ok = len(written) == len(verified)   # 書いた全部が検証通過
    summary = {
        "written": written, "verified": verified, "skipped": len(skipped),
        "effective_n": eff_n, "both_ends_rate": round(rate, 3),
        "raw_flb": raw_flb, "target_n": 300, "all_verified": all_ok,
        "backup_dir": BACKUP_DIR,
    }
    if verbose:
        print(f"export → {BACKUP_DIR}")
        print(f"  書込 {written} / 検証OK {verified} / 過去日skip {len(skipped)}")
        print(f"  有効N(FLB×両端10&1) = {eff_n}/300  両端揃い率={rate:.1%}  生FLB={raw_flb}")
        print(f"  全検証通過={all_ok}")
    # 検証が全通過したときだけ export心拍を打つ(=「保存されてる」の証明)
    if all_ok:
        _ping()
    return summary


if __name__ == "__main__":
    export()
