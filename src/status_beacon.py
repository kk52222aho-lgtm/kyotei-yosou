"""収集の生死を **リポジトリ経由で外へ出す** 死活灯台。healthchecks の URL が要らん版。

=== なぜ要るか ===
2026-09-05〜09-08、常駐が3日死んどった。watchdog は検知して **286回**再起動を試み、
286回とも失敗し、ログにも書いた。**やがそのログを読む者が居らんかった。**
`HC_ODDS_LOOP` / `HC_ODDS_EXPORT` / `HEALTHCHECK_DAILY` は3本とも未設定のまま。

**外へ出す経路が1本も無いのが本体の問題**やから、既に配線済みの通知経路に相乗りする:

    ローカル(毎朝) → status.json を GitHub API で直push
                  → GitHub Actions が読んで、鮮度が切れとったら **失敗する**
                  → GitHub が失敗メールを飛ばす(既定で有効・新しい登録は要らん)

**PC ごと落ちた場合も捕まる。**status.json が古いまま更新されんので、
クラウド側の判定が落ちる。watchdog(プロセス死しか見れん)では絶対に捕まらんケースや。

git の作業ツリーには一切触らん(Contents API で直接 PUT する)。
このリポは他の変更が乗ったままなので、`git add`/`commit` は事故のもとや。

usage:
  python -u -m src.status_beacon show     状態を出すだけ
  python -u -m src.status_beacon push     GitHub へ status.json を置く
  python -u -m src.status_beacon check    鮮度を判定(切れとったら exit 1)
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import sys

from . import storage, heartbeat

REPO = "kk52222aho-lgtm/kyotei-yosou"
PATH_IN_REPO = "data/status.json"

# 表: (名前, 表, 許容遅れ日数)。**収集の性質で許容が違う**
#   odds_timeseries は上書き型やない前向き収集=1日でも欠けたら永久欠損。厳しくする
TABLES = [
    ("entries", 3),
    ("payouts", 3),
    ("results_st", 3),
    ("odds_timeseries", 2),
    ("kachimake", 14),
    ("oriten", 14),
]
# 🚨 2026-09-09: 許容を**日**やのうて**時間**にして、仕事の周期に合わせた。
#   それまで全部 1〜2日で切っとったので、**10分ごとの watchdog が8時間黙っても ok**やった
#   (実際に PC が寝て 02:40→10:40 で8時間空いた)。
#   緩すぎる旗は無い旗と同じ([[insight_missing_reads_as_zero]] の裏返し)。
#   周期の目安 = 実行間隔 x 3 + 余裕。
LOGS = [
    # (ファイル, 許容時間, 実行周期の説明)
    ("watchdog.log", 3, "10分ごと。**稼働時間外も書く**ので一番鋭い生死signal"),
    ("probe_loop.log", 14, "稼働9-23時のみ毎分。夜間は書かんので長めに取る"),
    ("daily_collect.log", 30, "毎朝07:30"),
    ("pushout_record.log", 30, "毎朝08:30"),
    ("shrink_record.log", 30, "毎朝08:30"),
]

# 🚨 2026-09-09: 派生物の心拍を足す。
#   表とログの鮮度だけ見とったら、**cronが無い派生物**が静かに腐る。
#   実際 `data/model.joblib`(サイトが読む本番モデル)は **56日** 誰も作り直しとらんかった。
#   パイプライン自体は生きとる(クラウドが毎朝 today_picks.json を出しとる)のに、
#   その中に入っとるモデルが7月中旬のまま。**動いとるログでは絶対に見えん壊れ方や。**
#   判定は「新しいか」やのうて **派生物.mtime < 入力の最新日** かどうか。
#
#   ★凍結が正しいもんと区別する。事前登録で凍らせたモデルを毎日STALEと言うたら
#     誤報に慣れて死活装置ごと死ぬ。
DERIVED = [                      # (パス, 入力より何日古かったらSTALEか, 説明)
    ("model.joblib", 14, "本番モデル。サイトの予想はこれで出る"),
    ("motor_boat.csv", 60, "モーター/ボート番号。走査と今節特徴が使う"),
]
FROZEN = [                       # 凍結が正しい = 古くて当たり前。数えるが旗は立てん
    ("pushout_model.joblib", "事前登録で凍結(docs/prereg_pushout.md)"),
    ("envelope_357.csv", "②の固定抽出。動かしたら証人が壊れる"),
]


def collect() -> dict:
    today = dt.date.today()
    conn = storage.connect()
    tables = {}
    for name, tol in TABLES:
        try:
            mx, n = conn.execute(f"SELECT MAX(date), COUNT(*) FROM {name}").fetchone()
            lag = (today - dt.datetime.strptime(mx, "%Y%m%d").date()).days if mx else 999
            tables[name] = {"latest": mx, "rows": n, "lag_days": lag,
                            "tol_days": tol, "stale": lag > tol}
        except Exception as e:
            tables[name] = {"error": str(e), "stale": True}
    conn.close()

    logs = {}
    for fn, tol_h, note in LOGS:
        p = os.path.join(storage.DATA_DIR, fn)
        if not os.path.exists(p):
            logs[fn] = {"missing": True, "stale": True, "note": note}
            continue
        age = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(p)))
        h = age.total_seconds() / 3600
        logs[fn] = {"mtime": dt.datetime.fromtimestamp(os.path.getmtime(p))
                    .isoformat(timespec="minutes"),
                    "age_hours": round(h, 1), "tol_hours": tol_h,
                    "stale": h > tol_h, "note": note}

    # watchdog が「再起動に失敗し続けとる」のは、プロセス死より悪い状態や。
    # 🚨 2026-09-08: 直近60行を素で数えたら、**もう復旧した後の古い失敗**を拾って
    #    毎日 STALE を出しとった。誤報に慣れたら死活装置は死ぬ。
    #    → **最後に生きとった証(alive / restarted)より後の失敗だけ**を数える。
    wd = {"recent_restart_failures": 0, "last_line": None, "duplicate_loops": 0}
    p = os.path.join(storage.DATA_DIR, "watchdog.log")
    if os.path.exists(p):
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        tail = lines[-200:]
        last_ok = max((i for i, x in enumerate(tail)
                       if "alive" in x or "restarted" in x), default=-1)
        after = tail[last_ok + 1:]
        wd["recent_restart_failures"] = sum("RESTART FAILED" in x for x in after)
        wd["duplicate_loops"] = sum("DUPLICATE LOOPS" in x for x in after)
        wd["last_line"] = tail[-1] if tail else None
        wd["since_last_ok"] = len(after)

    # 派生物: 入力(entries の最新日)より古いか
    src_day = tables.get("entries", {}).get("latest")
    derived, frozen = {}, {}
    if src_day:
        sd = dt.datetime.strptime(src_day, "%Y%m%d")
        for fn, tol, note in DERIVED:
            fp = os.path.join(storage.DATA_DIR, fn)
            if not os.path.exists(fp):
                derived[fn] = {"missing": True, "stale": True, "note": note}
                continue
            m = dt.datetime.fromtimestamp(os.path.getmtime(fp))
            lag = (sd - m).days
            derived[fn] = {"built": m.strftime("%Y-%m-%d"), "lag_days": lag,
                           "tol_days": tol, "stale": lag > tol, "note": note}
        for fn, note in FROZEN:
            fp = os.path.join(storage.DATA_DIR, fn)
            if os.path.exists(fp):
                m = dt.datetime.fromtimestamp(os.path.getmtime(fp))
                frozen[fn] = {"built": m.strftime("%Y-%m-%d"),
                              "lag_days": (sd - m).days, "note": note}

    hb = {n: bool(heartbeat.url(n)) for n in (heartbeat.LOOP, heartbeat.EXPORT)}
    stale = ([k for k, v in tables.items() if v.get("stale")]
             + [k for k, v in logs.items() if v.get("stale")]
             + [k for k, v in derived.items() if v.get("stale")])
    if wd["recent_restart_failures"] >= 3:
        stale.append("watchdog:restart_failing")
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": today.strftime("%Y%m%d"),
        "tables": tables, "logs": logs, "watchdog": wd,
        "derived": derived, "frozen": frozen,
        "heartbeat_urls_set": hb,
        "stale": stale, "ok": not stale,
    }


def show(st: dict) -> None:
    print(f"generated_at {st['generated_at']}")
    print("\n  表                 最新        行数        遅れ  許容  状態")
    for k, v in st["tables"].items():
        if "error" in v:
            print(f"  {k:18s} ERROR {v['error'][:40]}")
            continue
        mark = "STALE" if v["stale"] else "ok"
        print(f"  {k:18s} {v['latest']}  {v['rows']:>10,}  {v['lag_days']:>4}d "
              f"{v['tol_days']:>4}d  {mark}")
    print("\n  ログ                 最終書込           経過      状態")
    for k, v in st["logs"].items():
        if v.get("missing"):
            print(f"  {k:20s} MISSING")
            continue
        mark = "STALE" if v["stale"] else "ok"
        print(f"  {k:20s} {v['mtime']}  {v['age_hours']:>6.1f}h  {mark}")
    if st.get("derived"):
        print("" + chr(10) + "  派生物(入力より古いか)      作られた日    遅れ  許容  状態")
        for k, v in st["derived"].items():
            if v.get("missing"):
                print(f"  {k:<26} MISSING")
                continue
            mark = "STALE" if v["stale"] else "ok"
            print(f"  {k:<26} {v['built']}  {v['lag_days']:>4}d {v['tol_days']:>4}d  {mark}"
                  f"   {v['note']}")
    if st.get("frozen"):
        print("  凍結(古くて正しい):")
        for k, v in st["frozen"].items():
            print(f"    {k:<24} {v['built']} ({v['lag_days']:+d}d)  {v['note']}")
    w = st["watchdog"]
    print(f"\n  watchdog: 直近60行に RESTART FAILED {w['recent_restart_failures']}回 / "
          f"DUPLICATE {w.get('duplicate_loops', 0)}回")
    print(f"            last: {w['last_line']}")
    print(f"  heartbeat URL: {st['heartbeat_urls_set']}")
    print(f"\n  => {'OK' if st['ok'] else 'STALE: ' + ', '.join(st['stale'])}")


def push(st: dict) -> None:
    """Contents API で status.json を直 PUT。作業ツリーには触らん。

    🚨 2026-09-08: `.env` の GITHUB_TOKEN は contents:write を持っとらん(403)。
       `gh` CLI の方は repo スコープ付きで認証済みなので、そっちを使う。
       トークンを新しく作らせるより、既にある権限に乗る方が壊れにくい。
    """
    import subprocess
    # 🚨 gh は環境変数 GH_TOKEN / GITHUB_TOKEN をキーリングより優先する。
    #    heartbeat.load_env() が .env の弱い PAT を os.environ に入れとるので、
    #    そのまま subprocess に渡すと gh がそっちを使うて 403 になる。**剥がして渡す。**
    env = {k: v for k, v in os.environ.items()
           if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
    body = json.dumps(st, ensure_ascii=False, indent=1).encode("utf-8")
    b64 = base64.b64encode(body).decode("ascii")
    api = f"/repos/{REPO}/contents/{PATH_IN_REPO}"

    sha = None
    r = subprocess.run(["gh", "api", api], capture_output=True, text=True, env=env)
    if r.returncode == 0:
        try:
            sha = json.loads(r.stdout).get("sha")
        except Exception:
            pass

    args = ["gh", "api", "--method", "PUT", api,
            "-f", f"message=status beacon {st['date']} "
                  f"({'ok' if st['ok'] else 'STALE'})",
            "-f", f"content={b64}"]
    if sha:
        args += ["-f", f"sha={sha}"]
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    if r.returncode == 0:
        print(f"push ok ({PATH_IN_REPO})")
    else:
        print(f"push FAILED rc={r.returncode}: {r.stderr.strip()[:300]}")
        sys.exit(1)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    st = collect()
    show(st)
    if cmd == "push":
        push(st)
    elif cmd == "check" and not st["ok"]:
        print("\nDEADMAN: 収集が止まっとる。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
