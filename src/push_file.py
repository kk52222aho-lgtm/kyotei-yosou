"""任意のファイルを Contents API でリポジトリへ直 PUT する。作業ツリーには触らん。

## なんで要るか

クラウド(GitHub Actions の `daily-scan.yml`)が毎朝 `src.scan` を回して
`today_picks.json` を作っとる。そこには **`data/kyotei.db` が無い**(gitignore)。
走査24本を積んだモデルを使うには、値の写し(`data/scan_recent.csv`)が
**クラウドが走る前に**リポに入っとらなあかん。

`status_beacon.push` と同じ道(gh CLI の Contents API)に乗る。トークンを
新しく作らせるより、既に権限が在る道に乗る方が壊れにくい。
作業ツリーを触らんのも同じ理由——このリポは他の未コミット変更を抱えとる。

usage:
    python -m src.push_file data/scan_recent.csv "scan features 20260927"
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

REPO = "kk52222aho-lgtm/kyotei-yosou"


def push(local: str, path_in_repo: str | None = None, message: str | None = None) -> int:
    if not os.path.exists(local):
        print(f"🚨 無い: {local}")
        return 1
    path_in_repo = path_in_repo or local.replace("\\", "/")
    message = message or f"push {path_in_repo}"
    # 🚨 gh は GH_TOKEN / GITHUB_TOKEN をキーリングより優先する。`.env` の弱い PAT が
    #    os.environ に入っとると 403 になるんで**剥がして渡す**(status_beacon と同じ)。
    env = {k: v for k, v in os.environ.items()
           if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
    with open(local, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    api = f"/repos/{REPO}/contents/{path_in_repo}"

    sha = None
    r = subprocess.run(["gh", "api", api], capture_output=True, text=True, env=env)
    if r.returncode == 0:
        try:
            sha = json.loads(r.stdout).get("sha")
        except Exception:
            pass
    # 🚨 `-f content=<base64>` やと **Windows のコマンド行上限(32,767字)**に当たる。
    #    45KB のファイルで `WinError 206 ファイル名または拡張子が長すぎます` が出た。
    #    `status_beacon` は小さい JSON やから通っとっただけ。**本体は標準入力から渡す。**
    body = {"message": message, "content": b64}
    if sha:
        body["sha"] = sha
    r = subprocess.run(["gh", "api", "--method", "PUT", api, "--input", "-"],
                       input=json.dumps(body), capture_output=True, text=True,
                       env=env, encoding="utf-8")
    if r.returncode != 0:
        print(f"push FAILED rc={r.returncode}: {r.stderr.strip()[:300]}")
        return 1
    n = os.path.getsize(local)
    print(f"push ok ({path_in_repo}, {n:,} bytes)")
    return 0


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    local = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(push(local, None, msg))


if __name__ == "__main__":
    main()
