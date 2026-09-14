"""押し出し検出器の前向き記録。凍結モデルで毎日スコアを付けて残す。

事前登録: docs/prereg_pushout.md(2026-09-03)。**再学習も閾値変更もせん。**

規律:
- **全レースを記録する。**発火だけ残したら、後から閾値を動かして解釈できてしまう
- 記録は締切前。**着順・払戻は一切見ずに保存する**
- 突合は別コマンド(`settle`)。確定払戻だけを使う

usage:
  python -u -m src.pushout_record log            本日ぶんを記録
  python -u -m src.pushout_record log 20260903   日付指定
  python -u -m src.pushout_record settle         未確定ぶんに確定払戻を突合
  python -u -m src.pushout_record report         中間集計(条項の基準は動かさん)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from . import storage, scraper
from .venues import name as venue_name

LEDGER = os.path.join(storage.DATA_DIR, "pushout_ledger.jsonl")
TODAY_JSON = os.path.join(storage.DATA_DIR, "pushout_today.json")
REPO = "kk52222aho-lgtm/kyotei-yosou"
MODEL = os.path.join(storage.DATA_DIR, "pushout_model.joblib")


# ---- 選手ごとの前づけ癖(その日より前の履歴だけ) --------------------------
def history(conn, before: str):
    """before より前の results_st から、選手×枠 / 選手 / 場 の率を作る。"""
    q = """
        SELECT s.jcd, s.lane, s.course, e.reg
        FROM results_st s JOIN entries e
          ON s.date=e.date AND s.jcd=e.jcd AND s.rno=e.rno AND s.lane=e.lane
        WHERE s.date < ?
    """
    d = pd.read_sql_query(q, conn, params=(before,))
    d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    d["reg"] = d["reg"].astype(str)
    d["grab1"] = (d["course"] == 1).astype(float)
    d["in_"] = (d["course"] < d["lane"]).astype(float)
    d["out1"] = ((d["lane"] == 1) & (d["course"] > 1)).astype(float)
    d["hold1"] = ((d["lane"] == 1) & (d["course"] == 1)).astype(float)
    g_rl = d.groupby(["reg", "lane"]).agg(n=("grab1", "size"),
                                          grab=("grab1", "mean"),
                                          inn=("in_", "mean"))
    g_r = d.groupby("reg").agg(n=("out1", "size"), out1=("out1", "mean"),
                               hold1=("hold1", "mean"))
    g_j = d.groupby("jcd")["out1"].mean()
    return g_rl, g_r, g_j


MIN_OBS = 5


def features_for(entries, g_rl, g_r, g_j, jcd):
    """出走表6艇から凍結モデルの7特徴を作る。履歴不足なら None。"""
    grabs, inns = [], []
    for e in entries:
        reg, lane = str(e.get("reg")), int(e.get("lane"))
        if lane == 1:
            continue
        r = g_rl.loc[(reg, lane)] if (reg, lane) in g_rl.index else None
        if r is None or r["n"] < MIN_OBS:
            return None
        grabs.append(r["grab"])
        inns.append(r["inn"])
    b1 = next((e for e in entries if int(e.get("lane")) == 1), None)
    if b1 is None or len(grabs) != 5:
        return None
    reg1 = str(b1.get("reg"))
    if reg1 not in g_r.index or g_r.loc[reg1, "n"] < MIN_OBS:
        return None
    if jcd not in g_j.index:
        return None
    return {
        "grab_max": float(np.max(grabs)), "grab_sum": float(np.sum(grabs)),
        "press_max": float(np.max(inns)), "press": float(np.sum(inns)),
        "b1_out": float(g_r.loc[reg1, "out1"]),
        "b1_hold": float(g_r.loc[reg1, "hold1"]),
        "v_out": float(g_j.loc[jcd]),
    }


def cmd_log(date: str):
    b = joblib.load(MODEL)
    m, FEA, thr = b["model"], b["features"], b["threshold"]
    print(f"凍結モデル {b['frozen_at']}  閾値 p>={thr:.6f}  "
          f"学習 {b['train_from']}-{b['train_to']}")
    conn = storage.connect()
    g_rl, g_r, g_j = history(conn, date)
    conn.close()
    print(f"履歴: 選手×枠 {len(g_rl):,}組 / 選手 {len(g_r):,}人 "
          f"({date} より前のみ)")

    seen = set()
    if os.path.exists(LEDGER):
        for ln in open(LEDGER, encoding="utf-8"):
            r = json.loads(ln)
            seen.add((r["date"], r["jcd"], r["rno"]))

    venues = scraper.fetch_held_venues(date)
    print(f"{date} 開催 {len(venues)}場: {' '.join(venue_name(j) for j in venues)}")
    n_new = n_fire = n_skip = 0
    with open(LEDGER, "a", encoding="utf-8") as f:
        for jcd in venues:
            for rno in range(1, 13):
                if (date, jcd, rno) in seen:
                    continue
                ent = scraper.fetch_racelist(date, jcd, rno)
                if not ent or len(ent) != 6:
                    continue
                fe = features_for(ent, g_rl, g_r, g_j, jcd)
                if fe is None:
                    n_skip += 1
                    continue
                p = float(m.predict_proba(
                    np.array([[fe[c] for c in FEA]], dtype=float))[0, 1])
                fire = p >= thr
                rec = {
                    "date": date, "jcd": jcd, "場": venue_name(jcd), "rno": rno,
                    "p": round(p, 6), "fire": bool(fire),
                    "feat": {k: round(v, 6) for k, v in fe.items()},
                    "regs": {int(e["lane"]): str(e.get("reg")) for e in ent},
                    "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "model_sha": b.get("frozen_at"),
                    "settled": False,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_new += 1
                if fire:
                    n_fire += 1
                    print(f"  🔥 {venue_name(jcd)}{rno}R  p={p:.4f}")
    print(f"\n記録 {n_new}レース / 発火 {n_fire} / 履歴不足でskip {n_skip}")
    print(f"台帳: {LEDGER}")
    publish(date, b, thr)


def publish(date, bundle, thr):
    """その日の発火レースを表示用JSONに出して、リポジトリへ押す。

    Streamlit は Community Cloud でリポジトリから動いとる。**DBはローカルにしか無い**
    (kyotei.db は gitignore)ので、クラウド側では前づけ癖を計算でけへん。
    → 判定結果だけを小さいJSONにして、status_beacon と同じ Contents API で押す。
    """
    import base64
    import subprocess
    rows = [r for r in read_ledger_rows() if r["date"] == date]
    fires = sorted([r for r in rows if r["fire"]], key=lambda r: -r["p"])
    payload = {
        "date": date,
        "generated_at": dt.datetime.now().isoformat(timespec="minutes"),
        "threshold": thr,
        "model_frozen_at": bundle.get("frozen_at"),
        "n_races": len(rows),
        "fires": [{"venue": r["場"], "rno": r["rno"], "p": r["p"],
                   "jcd": r["jcd"], "regs": r.get("regs", {})} for r in fires],
    }
    with open(TODAY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    # gh は環境変数のトークンをキーリングより優先する。弱いPATを剥がして渡す
    env = {k: v for k, v in os.environ.items()
           if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
    api = f"/repos/{REPO}/contents/data/pushout_today.json"
    r = subprocess.run(["gh", "api", api], capture_output=True, text=True, env=env)
    sha = None
    if r.returncode == 0:
        try:
            sha = json.loads(r.stdout).get("sha")
        except Exception:
            pass
    body = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
    args = ["gh", "api", "--method", "PUT", api,
            "-f", f"message=pushout {date} fires={len(fires)}",
            "-f", f"content={base64.b64encode(body).decode('ascii')}"]
    if sha:
        args += ["-f", f"sha={sha}"]
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    print(f"  出口JSON push {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr[:180]}"
          f"  (発火 {len(fires)}件 / {len(rows)}レース)")


def read_ledger_rows():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER, encoding="utf-8")]


def cmd_settle():
    if not os.path.exists(LEDGER):
        print("台帳が無い。先に log を。")
        return
    rows = [json.loads(ln) for ln in open(LEDGER, encoding="utf-8")]
    conn = storage.connect()
    pay = pd.read_sql_query(
        "SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
        "WHERE tansho_lane IS NOT NULL", conn)
    st = pd.read_sql_query(
        "SELECT date,jcd,rno,lane,course FROM results_st WHERE lane=1", conn)
    conn.close()
    for d in (pay, st):
        d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    P = pay.set_index(["date", "jcd", "rno"])
    S = st.set_index(["date", "jcd", "rno"])["course"]
    n = 0
    for r in rows:
        if r.get("settled"):
            continue
        k = (r["date"], r["jcd"], r["rno"])
        if k not in P.index:
            continue
        r["tansho_lane"] = int(P.loc[k, "tansho_lane"])
        r["tansho_yen"] = float(P.loc[k, "tansho_yen"])
        r["c1"] = int(S.loc[k]) if k in S.index else None
        r["settled"] = True
        n += 1
    with open(LEDGER, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"突合 {n}レース / 台帳 {len(rows)}レース")


def cmd_report():
    rows = [json.loads(ln) for ln in open(LEDGER, encoding="utf-8")]
    d = pd.DataFrame([r for r in rows if r.get("settled")])
    if not len(d):
        print(f"台帳 {len(rows)}レース、うち確定 0。まだ集計するもんが無い。")
        return
    print(f"確定 {len(d):,}レース / {d['date'].nunique()}日 "
          f"({d['date'].min()}-{d['date'].max()})")
    d["win1"] = (d["tansho_lane"] == 1).astype(int)
    d["out1"] = (d["c1"].fillna(1) > 1).astype(int)
    print("\n=== 事前登録の基準(docs/prereg_pushout.md) ===")
    for lab, sub in [("発火(p>=閾値)", d[d.fire]), ("非発火", d[~d.fire])]:
        if not len(sub):
            continue
        print(f"  {lab:<16}{len(sub):>6,}R  押し出し率 {sub['out1'].mean():>6.2%}"
              f"  1号艇勝率 {sub['win1'].mean():>6.2%}"
              f"  1号艇単勝回収 "
              f"{sub.loc[sub.win1 == 1, 'tansho_yen'].sum() / (len(sub) * 100):>6.1%}")
    f = d[d.fire]
    print(f"\n  N={len(f)}/300  "
          f"{'判定可' if len(f) >= 300 else 'まだ判定せん(条項どおり)'}")
    print("  ※ 単勝ROIはこの記録では決着せん(事前登録4節)。回収は記録するだけ")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "log"
    if cmd == "log":
        date = sys.argv[2] if len(sys.argv) > 2 else dt.date.today().strftime("%Y%m%d")
        cmd_log(date)
    elif cmd == "settle":
        cmd_settle()
    else:
        cmd_report()


if __name__ == "__main__":
    main()
