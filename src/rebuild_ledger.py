"""成績台帳を現行モデルで再生成する（＝再現記録／バックテスト）。

【重要】これは前向き記録ではない。過去のレースに後から現行モデルを当てた再現であり、
`rebuilt: true` を全行に付けて台帳の性格を機械的に区別できるようにする。
サイト側(成績ページ)はこのフラグを見て「再現記録(バックテスト)」と明示表示する。

再現できないもの:
  - 締切の単勝オッズ(final_odds)。過去分は取得不能なので None のまま。
    → 「単勝1.5倍未満は見送り」フィルタは再現できない = 無フィルタの記録として出す。
  - 締切間際の2連単EV(exacta_ev)。同上。
再現できるもの:
  - 買い目(現行モデル)と、確定払戻による全賭式の精算(payouts テーブル)。

実行: python -m src.rebuild_ledger --start 20260701 --end 20260806
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3

from . import predict, storage
from .scan import venue_name

LEDGER = os.path.join(storage.DATA_DIR, "papertrade.jsonl")
COLS = ("lane, reg, racer_class, name, age, weight, nat_win, nat_2rate, loc_win, "
        "loc_2rate, motor_2rate, boat_2rate, tenji_time, wind_speed, wave_height")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=LEDGER)
    a = ap.parse_args()

    bundle = predict.load_model()
    if bundle is None:
        raise SystemExit("モデルが無い")
    con = sqlite3.connect(f"file:{storage.DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    races = con.execute(
        "SELECT DISTINCT e.date, e.jcd, e.rno FROM entries e "
        "JOIN payouts p ON p.date=e.date AND p.jcd=e.jcd AND p.rno=e.rno "
        "WHERE e.date>=? AND e.date<=? AND p.tansho_yen IS NOT NULL "
        "ORDER BY e.date, e.jcd, e.rno", (a.start, a.end)).fetchall()
    print(f"対象レース {len(races):,} ({a.start}-{a.end})", flush=True)

    rows, skipped, nbet = [], 0, 0
    for i, r in enumerate(races):
        date, jcd, rno = r["date"], r["jcd"], r["rno"]
        ents = [dict(x) for x in con.execute(
            f"SELECT {COLS} FROM entries WHERE date=? AND jcd=? AND rno=? ORDER BY lane",
            (date, jcd, rno)).fetchall()]
        if len(ents) != 6:
            skipped += 1
            continue
        for e in ents:
            e["jcd"] = jcd
        try:
            pr = predict.predict_entries(ents, bundle)
            rec = predict.recommend(pr)
        except Exception:
            skipped += 1
            continue
        if not rec.get("bet"):
            continue
        nbet += 1
        pay = con.execute(
            "SELECT tansho_lane, tansho_yen, exacta_combo, exacta_yen, trio_combo, "
            "trio_yen, trifecta_combo, trifecta_yen FROM payouts "
            "WHERE date=? AND jcd=? AND rno=?", (date, jcd, rno)).fetchone()
        honmei = rec["tansho"]
        winner = pay["tansho_lane"]
        won = (winner == honmei)
        ex3 = rec.get("exacta3") or []
        tr4 = rec.get("trio4") or []
        tf3 = rec.get("trifecta3") or []
        ex_hit = pay["exacta_combo"] in ex3
        tr_hit = pay["trio_combo"] in tr4
        tf_hit = pay["trifecta_combo"] in tf3
        rows.append({
            "date": date, "jcd": jcd, "rno": rno, "venue": venue_name(jcd),
            "honmei": honmei, "name": rec.get("tansho_name"),
            "scan_odds": None, "win_pct": pr[0].get("win_pct"),
            "ichi_pct": next((x["win_pct"] for x in pr if x["lane"] == 1), None),
            "exacta3": ex3, "exacta3_p": rec.get("exacta3_p"),
            "trio4": tr4, "trio4_p": rec.get("trio4_p"),
            "trifecta3": tf3, "trifecta3_p": rec.get("trifecta3_p"),
            "trio_rank": rec.get("trio_rank"), "trifecta_rank": rec.get("trifecta_rank"),
            "deadline": None, "exacta_ev": None,
            "logged_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "settled": True, "rebuilt": True,           # ← 再現記録の印
            "winner": winner, "final_odds": None,        # 過去の締切オッズは再現不能
            "tansho_win": won, "tansho_return": int(pay["tansho_yen"]) if won else 0,
            "exacta_result": pay["exacta_combo"], "exacta_points": len(ex3),
            "exacta_win": ex_hit, "exacta_return": int(pay["exacta_yen"] or 0) if ex_hit else 0,
            "trio_result": pay["trio_combo"], "trio_points": len(tr4),
            "trio_win": tr_hit, "trio_return": int(pay["trio_yen"] or 0) if tr_hit else 0,
            "trifecta_result": pay["trifecta_combo"], "trifecta_points": len(tf3),
            "trifecta_win": tf_hit,
            "trifecta_return": int(pay["trifecta_yen"] or 0) if tf_hit else 0,
            "res_full": {
                "winner": winner,
                "tansho_yen": int(pay["tansho_yen"]) if won else 0,
                "exacta_combo": pay["exacta_combo"], "exacta_yen": pay["exacta_yen"],
                "trio_combo": pay["trio_combo"], "trio_yen": pay["trio_yen"],
                "trifecta_combo": pay["trifecta_combo"], "trifecta_yen": pay["trifecta_yen"],
            },
        })
        if i % 500 == 0:
            print(f"  {i}/{len(races)} 勝負{nbet}", flush=True)
    con.close()

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    tw = sum(1 for r in rows if r["tansho_win"])
    tr_ = sum(r["tansho_return"] for r in rows)
    print(f"\n再生成 {n:,}行（勝負レース）/ スキップ {skipped}")
    print(f"  単勝: 的中 {tw}/{n} ({tw/n*100:.1f}%) 回収 {tr_/(n*100)*100:.1f}%")
    for k, pts in (("exacta", 3), ("trio", 4), ("trifecta", 3)):
        h = sum(1 for r in rows if r[f"{k}_win"])
        rr = sum(r[f"{k}_return"] for r in rows)
        print(f"  {k}: 的中 {h}/{n} ({h/n*100:.1f}%) 回収 {rr/(n*pts*100)*100:.1f}%")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
