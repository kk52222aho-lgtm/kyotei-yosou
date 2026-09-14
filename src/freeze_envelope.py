"""②オッズ動態の凍結封筒を **不変の固定抽出** に落とす(開封の前工程)。

=== なぜ固定するか ===
probe_loop を再起動すると kyotei.db への書込が再開する。判定を生DBに対して回すと
「開封作業中にDBが動いとった」いう可能性が残る。**判定対象を先に凍らせてから
常駐を起こす。**これで順序の問題が最初から消える。

=== やること(ROI・着順は一切見ん) ===
1. SQLite の **backup API** で DB 全体のスナップショット(ファイルコピーやのうて
   トランザクション整合の取れた複製。書込中でも壊れん)
2. スナップショットから②条項の有効レースを抜いて `data/envelope_357.csv` に固定
   有効の定義(export_odds._effective_n と同一):
     - bet_type='tansho'
     - **FLB**: 本命(honmei)が1号艇やない
     - **両端**: その本命艇が mins_to_deadline = 10 と 1 の両方を持っとる
3. 抽出の指紋(件数・SHA256・日付範囲)を出す

**開封の判定はこの CSV に対してだけ回す。生DBには当てん。**

usage: python -u -m src.freeze_envelope
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sqlite3

import pandas as pd

from . import storage

SNAP_DIR = os.path.join(storage.DATA_DIR, "snapshots")
OUT = os.path.join(storage.DATA_DIR, "envelope_357.csv")


def snapshot() -> str:
    os.makedirs(SNAP_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(SNAP_DIR, f"kyotei_{stamp}.db")
    src = sqlite3.connect(storage.DB_PATH if hasattr(storage, "DB_PATH")
                          else os.path.join(storage.DATA_DIR, "kyotei.db"))
    out = sqlite3.connect(dst)
    with out:
        src.backup(out)          # backup API: 書込中でも整合の取れた複製
    out.close()
    src.close()
    print(f"スナップショット: {dst}  ({os.path.getsize(dst) / 1e6:.0f} MB)")
    return dst


def freeze(db: str):
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT date, jcd, rno, honmei, combo, mins_to_deadline "
        "FROM odds_timeseries WHERE bet_type='tansho'").fetchall()
    honmei, hon_mins = {}, {}
    for date, jcd, rno, hon, combo, mins in rows:
        k = (date, jcd, rno)
        if hon is not None:
            honmei[k] = hon
            if str(combo) == str(hon):
                hon_mins.setdefault(k, set()).add(mins)
    flb = [k for k, h in honmei.items() if h != 1]
    eff = sorted(k for k in flb if {10, 1} <= hon_mins.get(k, set()))
    print(f"生FLB(本命≠1号) = {len(flb)}   有効(両端10&1そろい) = {len(eff)}")

    keys = pd.DataFrame(eff, columns=["date", "jcd", "rno"])
    ts = pd.read_sql_query("SELECT * FROM odds_timeseries", conn)
    conn.close()
    sub = ts.merge(keys, on=["date", "jcd", "rno"], how="inner")
    sub = sub.sort_values(["date", "jcd", "rno", "bet_type", "combo",
                           "mins_to_deadline"]).reset_index(drop=True)
    sub.to_csv(OUT, index=False)

    h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print(f"\n=== 固定抽出 {OUT} ===")
    print(f"  レース {len(keys):,} / 行 {len(sub):,}")
    print(f"  日付   {sub['date'].min()} - {sub['date'].max()}")
    print(f"  窓     {sorted(sub['mins_to_deadline'].unique())}")
    print(f"  券種   {sorted(sub['bet_type'].unique())}")
    print(f"  SHA256 {h}")
    print("\n  ※ 着順・払戻・ROI は一切触っとらん。開封の判定はこのCSVに対してだけ回す。")
    keys.to_csv(os.path.join(storage.DATA_DIR, "envelope_357_keys.csv"), index=False)


def main():
    db = snapshot()
    freeze(db)


if __name__ == "__main__":
    main()
