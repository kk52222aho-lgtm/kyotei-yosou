"""走査の選抜を **pit版の窓(前日まで)** でやり直す。

## なんで要るか

本番に積んだ24本は `data/scan_holdout.csv` から取っとる。その選抜は
**同じ日の情報が入った計算**(`scan_all.lagged`)で行われた。値の方は pit で計算し直して
skew をゼロにしたが、**どの24本を選ぶかは同日情報つきのまま**や。台帳にも
「選抜のバイアスは残っとる」と書いて残してある。ここを潰す。

## 変えるんは1つだけ

窓の計算を `scan_pit.lagged_pit` に差し替えるだけで、基準の残差(`data/oof_base.csv`)も
cutoff も閾値も**そのまま**。2つ動かしたら「ちがいが出た原因」が分かれんくなる
→ [[insight_control_group_design]]。

`scan_all.py` は**書き換えん**(保存済みの `scan_all.csv` / `scan_holdout.csv` が
選抜の証人。書き換えたら比べられんくなる)。module 直下の `lagged` を
外から差し替えるだけにしてある。

出力は `data/scan_holdout_pit.csv`。元のファイルは触らん。

usage: python -u -m src.scan_pit_select
"""
from __future__ import annotations

from . import scan_all
from .scan_pit import lagged_pit

OUT = "data/scan_holdout_pit.csv"


def main() -> None:
    print("窓を pit版(その日より前)に差し替えて選抜し直す")
    print(f"  基準残差: data/oof_base.csv(そのまま) / cutoff=2025 / 出力={OUT}")
    scan_all.lagged = lagged_pit          # 🚨 ここ1つだけ変える
    df = scan_all.load()
    scan_all.scan(df, cutoff="2025", out=OUT)


if __name__ == "__main__":
    main()
