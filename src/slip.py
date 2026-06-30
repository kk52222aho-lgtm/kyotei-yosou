"""買い目スリップ生成（半自動運用）。

本日の妙味レース（scan のキャッシュ）から、テレボートに手入力しやすい
コピペ用の買い目リストを作る。購入は自分の手で（規約・金銭事故リスク回避）。

戦略: 各妙味レースで 単勝1点(本命) ＋ 2連単3点(確率上位) を少額で。

例:
  python -m src.slip            # 本日のキャッシュからスリップ表示
  python -m src.slip --unit 200 # 1点200円で
"""
from __future__ import annotations

import argparse

from .scan import load_cache


def format_slip(cache: dict | None, unit: int = 100) -> str:
    if not cache:
        return "まだスキャンしていません（python -m src.scan）。"
    date = cache["date"]
    picks = cache["picks"]
    if not picks:
        return f"{date} 本日の妙味レースなし＝全レース見送り推奨。"

    lines = [f"━━ {date} 本日の買い目（1点 {unit}円・少額分散）━━"]
    total = 0
    for p in picks:
        ex = " ".join(p["exacta3"])
        lines.append(
            f"◆ {p['venue']} {p['rno']}R")
        lines.append(f"   単勝  : {p['honmei']}")
        lines.append(f"   2連単 : {ex}")
        total += unit * (1 + len(p["exacta3"]))   # 単勝1点 + 2連単3点
    n = len(picks)
    lines.append(f"━━ 計 {n}レース ／ 賭金合計 {total:,}円 ━━")
    lines.append("※テレボートで各レース「式別」を選んで上記を入力。")
    lines.append("※1日の勝敗に一喜一憂せず、数を重ねて平均で取る戦略。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", type=int, default=100, help="1点あたりの賭け額(円)")
    args = ap.parse_args()
    print(format_slip(load_cache(), args.unit))


if __name__ == "__main__":
    main()
