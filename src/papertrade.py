"""前向きペーパートレード台帳（最後の穴＝実弾妥当性を実記録で検証）。

前進検証(validate.py)は過去データで通った。残る穴は「実際に賭けられるか」だけ。
競艇はパリミュチュエル＝いつ賭けても払戻は確定オッズなので、本戦略(オッズでなく
モデル信号で引く)なら払戻＝確定 tansho。残る現実リスクは「自分の賭けでプールが
動くか」のみ＝少額なら無視可。それでも"紙の上"でなく前向き実記録で確かめる:

  1. log    : 締切前に本日の妙味レースの買い目＋その時点オッズを記録（結果を見る前）
  2. settle : 確定後、実結果＋確定払戻で精算（単勝フラット100円）
  3. report : 精算済みだけの実トラック回収率

※ log は「結果が出る前」に走らせて初めて前向き検証になる。過去日での log→settle は
  機能テスト（前向き記録ではない）。

例:
  python -m src.papertrade log              # 本日ぶんを記録（毎朝・締切前に）
  python -m src.papertrade settle           # 確定したレースを精算
  python -m src.papertrade report
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

from . import predict, scraper, storage
from .venues import name as venue_name

LEDGER = os.path.join(storage.DATA_DIR, "papertrade.jsonl")


def _load() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _save(rows: list[dict]) -> None:
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _key(r):
    return (r["date"], r["jcd"], r["rno"])


def cmd_log(date: str):
    """締切前: 妙味レースの買い目＋現オッズを記録（重複はスキップ）。"""
    bundle = predict.load_model()
    rows = _load()
    seen = {_key(r) for r in rows}
    venues = scraper.fetch_held_venues(date)
    print(f"{date} 開催{len(venues)}場をスキャン（締切前オッズを記録）")
    added = 0
    import time
    for jcd in venues:
        empty = 0
        for rno in range(1, 13):
            entries = scraper.fetch_racelist(date, jcd, rno)
            time.sleep(0.4)
            if not entries:
                empty += 1
                if empty >= 2:
                    break
                continue
            empty = 0
            if (date, jcd, rno) in seen:
                continue
            for e in entries:
                e["jcd"] = jcd
            pred = predict.predict_entries(entries, bundle)
            rec = predict.recommend(pred)
            if not rec.get("bet"):
                continue
            o = scraper.fetch_odds(date, jcd, rno)
            scan_odds = o.get(rec["tansho"]) if o else None
            rows.append({
                "date": date, "jcd": jcd, "rno": rno,
                "venue": venue_name(jcd), "honmei": rec["tansho"],
                "name": rec.get("tansho_name"), "scan_odds": scan_odds,
                "exacta3": rec["exacta3"],
                "logged_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "settled": False,
            })
            added += 1
            print(f"  記録 {venue_name(jcd)}{rno}R 単勝{rec['tansho']}号 "
                  f"({scan_odds}倍時点)")
    _save(rows)
    print(f"\n{added}件を記録（台帳: {LEDGER}）")


def cmd_settle(today: str):
    """確定済みレースを精算（単勝フラット100円。確定オッズ=払戻として実払戻計算）。"""
    rows = _load()
    n = 0
    for r in rows:
        if r.get("settled"):
            continue
        if r["date"] >= today:   # 当日以降は未確定とみなす
            continue
        result = scraper.fetch_result(r["date"], r["jcd"], r["rno"])
        if not result:
            continue
        winner = next((ln for ln, rank in result.items() if rank == 1), None)
        # 確定単勝オッズ（≈払戻/100）
        o = scraper.fetch_odds(r["date"], r["jcd"], r["rno"])
        final_odds = o.get(r["honmei"]) if o else None
        won = (winner == r["honmei"])
        r["winner"] = winner
        r["final_odds"] = final_odds
        r["tansho_win"] = won
        r["tansho_return"] = int(round((final_odds or 0) * 100)) if won else 0
        r["settled"] = True
        n += 1
    _save(rows)
    print(f"{n}件を精算")


def cmd_report():
    rows = [r for r in _load() if r.get("settled")]
    if not rows:
        print("精算済みの記録がまだありません（log→（結果後）settle）。")
        return
    bets = len(rows)
    hits = sum(1 for r in rows if r.get("tansho_win"))
    ret = sum(r.get("tansho_return", 0) for r in rows)
    stake = bets * 100
    print(f"=== ペーパートレード実トラック（単勝フラット100円）===")
    print(f"  賭けたレース: {bets}")
    print(f"  的中: {hits}（的中率 {hits/bets:.1%}）")
    print(f"  回収率: {ret/stake:.1%}   収支: {ret-stake:+,}円")
    print(f"\n  ※前向き記録（log を締切前に走らせたぶん）だけが本物の実トラック。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log"); lg.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    sub.add_parser("settle")
    sub.add_parser("report")
    args = ap.parse_args()
    today = dt.date.today().strftime("%Y%m%d")
    if args.cmd == "log":
        cmd_log(args.date)
    elif args.cmd == "settle":
        cmd_settle(today)
    elif args.cmd == "report":
        cmd_report()


if __name__ == "__main__":
    main()
