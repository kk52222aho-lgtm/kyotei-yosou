"""R5「縮」の前向き記録。事前登録 = docs/prereg_shrink.md(2026-09-08)。

=== N を 0 から積む理由 ===
20260716-20260905 の 392件は**探索に使うた**(あれで向きを見つけて条項を書いた)。
同じデータで採点したら自己採点や([[feedback_in_sample_rule_scoring]])。
→ **START から後だけを数える。**それ以前は1件も入れん。

=== 条項(凍結。動かさん) ===
  発火: モデル本命が1号艇やない(FLB) かつ その本命艇に T-10 と T-1 の両方がある
  群:   move = log( O(T-1) / O(T-10) )   move < 0 → **縮** / move >= 0 → 広
  買い: その本命艇の単勝1点、T-1 のタイミング
  対照: 同じレースの1号艇単勝(**予測が要らん手**)

  主要評価: **縮 − 広 の回収率差**。差のCIが0を跨がんこと。N=850/群(約150日)
  副次   : 縮の絶対ROI。CI下限≥100%。**N≈13,800=約7年で事実上不可能**と先に書いてある
  kill   : 縮の絶対ROI が N≥600 で 90% を割る

**収集が止まった日は「欠測」として記録する。**0件と未収集を同じ箱に入れん
([[insight_zero_is_a_measurement]])。

usage:
  python -u -m src.shrink_record log       今日までの未記録日を積む(着順は見ん)
  python -u -m src.shrink_record settle    確定払戻を突合
  python -u -m src.shrink_record report    中間集計(条項の基準は動かさん)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from . import storage

START = "20260909"          # 事前登録の翌日から。**動かさん**
LEDGER = os.path.join(storage.DATA_DIR, "shrink_ledger.jsonl")
N_MAIN = 850                # 主要評価に要る群あたりN
N_KILL = 600
RNG = np.random.default_rng(20260908)
NBOOT = 4000


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER, encoding="utf-8")]


def cmd_log():
    conn = storage.connect()
    d = pd.read_sql_query(
        "SELECT date,jcd,rno,combo,mins_to_deadline m,odds,honmei "
        "FROM odds_timeseries WHERE bet_type='tansho' AND date>=?",
        conn, params=(START,))
    conn.close()
    if not len(d):
        print(f"{START} 以降の収集がまだ無い。")
        return
    d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    d["lane"] = pd.to_numeric(d["combo"], errors="coerce")
    seen = {(r["date"], r["jcd"], r["rno"]) for r in read_ledger()}

    w = d.pivot_table(index=["date", "jcd", "rno", "lane"], columns="m",
                      values="odds")
    hon = (d[d.m == 1].groupby(["date", "jcd", "rno"])["honmei"].first())
    n_new = n_shrink = n_wide = n_skip = 0
    with open(LEDGER, "a", encoding="utf-8") as f:
        for (date, jcd, rno), h in hon.items():
            if (date, jcd, rno) in seen:
                continue
            if pd.isna(h) or int(h) == 1:          # FLB やない = 発火せん
                continue
            k = (date, jcd, rno, float(int(h)))
            if k not in w.index:
                n_skip += 1
                continue
            row = w.loc[k]
            o10, o1 = row.get(10), row.get(1)
            if pd.isna(o10) or pd.isna(o1):        # 両端がそろわん = 発火せん
                n_skip += 1
                continue
            move = float(np.log(o1 / o10))
            grp = "縮" if move < 0 else "広"
            f.write(json.dumps({
                "date": date, "jcd": jcd, "rno": int(rno),
                "honmei": int(h), "o10": float(o10), "o1": float(o1),
                "move": round(move, 6), "grp": grp,
                "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
                "settled": False,
            }, ensure_ascii=False) + "\n")
            n_new += 1
            n_shrink += grp == "縮"
            n_wide += grp == "広"
    print(f"記録 {n_new}件(縮 {n_shrink} / 広 {n_wide})  "
          f"両端そろわずskip {n_skip}")
    # 欠測日を明示する(0件と未収集を分ける)
    have = set(d["date"].unique())
    days = pd.date_range(START, dt.date.today().strftime("%Y%m%d"))
    miss = [x.strftime("%Y%m%d") for x in days
            if x.strftime("%Y%m%d") not in have]
    if miss:
        print(f"  ⚠ 収集が無い日 {len(miss)}日: {miss[:8]}{'…' if len(miss) > 8 else ''}")
        print("    (0件やのうて未収集。判定の分母から外れる)")


def cmd_settle():
    rows = read_ledger()
    if not rows:
        print("台帳が空。")
        return
    conn = storage.connect()
    pay = pd.read_sql_query("SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
                            "WHERE tansho_lane IS NOT NULL", conn)
    conn.close()
    pay["jcd"] = pay["jcd"].astype(str).str.zfill(2)
    P = pay.set_index(["date", "jcd", "rno"])
    n = 0
    for r in rows:
        if r.get("settled"):
            continue
        k = (r["date"], r["jcd"], r["rno"])
        if k not in P.index:
            continue
        wl = int(P.loc[k, "tansho_lane"])
        yen = float(P.loc[k, "tansho_yen"])
        r["won"] = int(wl == r["honmei"])
        r["ret"] = yen if r["won"] else 0.0
        r["lane1_won"] = int(wl == 1)          # 予測が要らん手
        r["lane1_ret"] = yen if wl == 1 else 0.0
        r["settled"] = True
        n += 1
    with open(LEDGER, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"突合 {n}件 / 台帳 {len(rows)}件")


def _ci(ret, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    s = np.bincount(iv, weights=ret, minlength=len(ud))
    c = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    return np.percentile(s[idx].sum(1) / (c[idx].sum(1) * 100), [2.5, 97.5])


def cmd_report():
    rows = [r for r in read_ledger() if r.get("settled")]
    if not rows:
        print(f"台帳 {len(read_ledger())}件、確定 0。まだ集計するもんが無い。")
        return
    D = pd.DataFrame(rows)
    print(f"確定 {len(D)}件 / {D['date'].nunique()}日 "
          f"({D['date'].min()}-{D['date'].max()})  事前登録 docs/prereg_shrink.md")
    print(f"\n  {'群':<6}{'N':>6}{'的中率':>9}{'回収率':>9}{'95%CI':>20}")
    stats = {}
    for g, s in D.groupby("grp"):
        ret = s["ret"].to_numpy(float)
        roi = ret.sum() / (len(s) * 100)
        lo, hi = _ci(ret, s["date"].to_numpy())
        stats[g] = (len(s), roi)
        print(f"  {g:<6}{len(s):>6,}{s['won'].mean():>9.2%}{roi:>9.1%}"
              f"  [{lo:>6.1%},{hi:>6.1%}]")
    r1 = D["lane1_ret"].to_numpy(float)
    print(f"  {'1号艇':<6}{len(D):>6,}{D['lane1_won'].mean():>9.2%}"
          f"{r1.sum() / (len(D) * 100):>9.1%}   ← 予測が要らん手")

    if "縮" in stats and "広" in stats:
        a = D[D.grp == "縮"]
        b = D[D.grp == "広"]
        diff = stats["縮"][1] - stats["広"][1]
        # 差の日クラスタ bootstrap
        ud = np.union1d(a["date"].unique(), b["date"].unique())
        ia = pd.Series(np.arange(len(ud)), index=ud)
        boot = []
        for _ in range(NBOOT):
            pick = RNG.integers(0, len(ud), len(ud))
            ds = ud[pick]
            ra = a[a.date.isin(ds)]
            rb = b[b.date.isin(ds)]
            if len(ra) and len(rb):
                boot.append(ra["ret"].sum() / (len(ra) * 100)
                            - rb["ret"].sum() / (len(rb) * 100))
        lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
        v = ("縮が上" if lo > 0 else ("広が上" if hi < 0 else "0を跨いどる"))
        print(f"\n  **主要: 縮 − 広 = {diff * 100:+.2f}pt  "
              f"CI[{lo * 100:+.2f},{hi * 100:+.2f}]  ← {v}**")
        n_min = min(stats['縮'][0], stats['広'][0])
        print(f"  N={n_min}/{N_MAIN}(群あたり)  "
              f"{'判定可' if n_min >= N_MAIN else 'まだ判定せん(条項どおり)'}")
        if stats["縮"][0] >= N_KILL and stats["縮"][1] < 0.90:
            print("  🚨 kill条項に該当(縮のROIが N≥600 で 90%割れ)")
    print("\n  ※ 副次(縮の絶対ROI)は N≈13,800=約7年で事実上決着せん。記録するだけ")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "log"
    {"log": cmd_log, "settle": cmd_settle}.get(cmd, cmd_report)()


if __name__ == "__main__":
    main()
