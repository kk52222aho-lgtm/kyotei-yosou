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

    FLOOR = 1.5  # 単勝オッズ下限（低オッズは控除率負け＝クリーン年検証済）
    lines = [f"━━ {date} 本日の買い目（1点 {unit}円・少額分散）━━"]
    total = 0
    for p in picks:
        ex = " ".join(p["exacta3"])
        o = p.get("odds") or p.get("scan_odds")
        lines.append(f"◆ {p['venue']} {p['rno']}R")
        if o and o < FLOOR:
            lines.append(f"   単勝  : {p['honmei']}（{o:.1f}倍→低オッズ・締切1.5倍未満なら見送り）")
        else:
            od = f"（{o:.1f}倍）" if o else ""
            lines.append(f"   単勝  : {p['honmei']}{od}")
            total += unit
        lines.append(f"   2連単 : {ex}")
        total += unit * len(p["exacta3"])
    n = len(picks)
    lines.append(f"━━ 計 {n}レース ／ 賭金 最大{total:,}円 ━━")
    lines.append("※単勝は締切オッズ1.5倍未満なら見送り（低オッズは控除率で負ける）。")
    lines.append("※1日の勝敗でなく数十〜百本の平均で判断。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", type=int, default=100, help="1点あたりの賭け額(円)")
    args = ap.parse_args()
    print(format_slip(load_cache(), args.unit))


if __name__ == "__main__":
    main()
