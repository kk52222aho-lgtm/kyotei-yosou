"""心拍の配線を1か所にまとめる。**黙って省略せん**版。

=== なぜ書き直したか(2026-09-02) ===
2026-08-03 の事故調で「心拍は一度も打たれたことがない」と分かって、healthchecks 側には
07-25 に登録した。**それでも 08-19 の全停止は手作業の点検で見つかった。**理由:

  - `C:\\dev\\.env` に `HC_ODDS_LOOP` / `HC_ODDS_EXPORT` は **無い**
  - kyotei-yosou は **dotenv を一切読まん**(他プロジェクトは `C:\\dev\\.env` 共有)

→ **URLがコードに届く経路が最初から存在してへんかった。**

=== 🚨 2026-09-08: この実装が常駐を殺した ===
上の対策として「未設定なら黙らず吠える」を足した。その警告文に **絵文字**を入れた。
`run_loop.bat` は `PYTHONIOENCODING` を立てんので stdout は **cp932**。
起動時の `announce()` が `UnicodeEncodeError` で落ち、**ループが起動即死**した。
watchdog は正しく検知して **286回**再起動を試み、286回とも同じ所で死んだ。
外部の目撃者(healthchecks)が繋がっとらんので、**3日間(9/05 16:20〜9/08)誰も気付かん**。

**沈黙検知のために足した「大声」が、その大声で常駐を殺した。**

→ **この模組の print は全部 ASCII のみ。**日本語はコメントと docstring だけに置く
  (それらは印字されん)。加えて `run_loop.bat` 側でも UTF-8 を立てて二重に守る。

usage:
  python -m src.heartbeat check     2本ともpingしてHTTPコードを出す(着弾の目視)
  python -m src.heartbeat show      設定の見え方だけ確認(pingせん)
"""
from __future__ import annotations

import os
import sys
import time

ENV_PATHS = [r"C:\dev\.env",
             os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                 os.path.abspath(__file__)))), ".env")]

LOOP = "HC_ODDS_LOOP"
EXPORT = "HC_ODDS_EXPORT"

_loaded = False
_warned_at: dict[str, float] = {}
WARN_EVERY = 600.0          # 秒: 未設定の警告を出し直す間隔


def load_env() -> None:
    """C:\\dev\\.env を読んで os.environ へ入れる(既存の環境変数は上書きせん)。"""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for p in ENV_PATHS:
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8-sig") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception as e:
            print(f"[heartbeat] failed to read .env {p}: {e}", flush=True)
        break


def url(name: str) -> str | None:
    load_env()
    return os.environ.get(name) or None


def ping(name: str) -> None:
    """心拍を1発打つ。未設定なら黙らんと警告を出す(ASCIIのみ)。"""
    u = url(name)
    if not u:
        now = time.monotonic()
        if now - _warned_at.get(name, -1e9) > WARN_EVERY:
            _warned_at[name] = now
            print(f"[heartbeat][WARN] {name} is NOT set - silence detection is DEAD. "
                  f"Add {name}=<healthchecks ping URL> to C:\\dev\\.env", flush=True)
        return
    try:
        import requests
        requests.get(u, timeout=10)
    except Exception:
        pass          # 心拍失敗そのものは握りつぶす(監視の監視はせん)


def announce() -> None:
    """起動時に1回、心拍の配線状態をログへ書く。**常駐が必ず通る道。ASCIIのみ。**"""
    load_env()
    for name in (LOOP, EXPORT):
        u = os.environ.get(name)
        if u:
            print(f"[heartbeat] {name} = set ({u[:28]}...)", flush=True)
        else:
            print(f"[heartbeat][WARN] {name} is NOT set - silence detection is DEAD.",
                  flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    load_env()
    found = [p for p in ENV_PATHS if os.path.exists(p)]
    print(f".env searched: {found or 'none'}")
    for name in (LOOP, EXPORT):
        u = os.environ.get(name)
        if not u:
            print(f"  {name:16s} NOT SET  <- no ping can ever land")
            continue
        print(f"  {name:16s} {u}")
        if cmd == "check":
            try:
                import requests
                r = requests.get(u, timeout=10)
                mark = "OK (landed)" if r.status_code == 200 else "FAILED"
                print(f"    -> HTTP {r.status_code} {mark}  "
                      f"(confirm the dot turns green on healthchecks.io)")
            except Exception as e:
                print(f"    -> failed: {e}")
    if cmd != "check":
        print("\nTo actually ping and see it land: python -m src.heartbeat check")


if __name__ == "__main__":
    main()
