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
    """本日の妙味レースの買い目を記録（重複はスキップ）。

    scan のキャッシュ(today_picks.json)から記録する＝あなたが画面で見て賭けた
    買い目と完全一致。再スキャンしないので速く、ズレも出ない。
    （毎朝の運用では run_daily が先に scan→キャッシュ更新してから log する）
    """
    from .scan import load_cache
    cache = load_cache()
    if not cache or cache.get("date") != date:
        print(f"{date} のキャッシュが無い。先に python -m src.scan --date {date} を実行。")
        return
    rows = _load()
    seen = {_key(r) for r in rows}
    added = 0
    for p in cache["picks"]:
        key = (p["date"] if "date" in p else date, p["jcd"], p["rno"])
        if key in seen:
            continue
        rows.append({
            "date": date, "jcd": p["jcd"], "rno": p["rno"],
            "venue": p["venue"], "honmei": p["honmei"],
            "name": p.get("name"), "scan_odds": p.get("odds"),
            "win_pct": p.get("win_pct"),   # 本命の正規化勝率(💎最強妙味判定 p0用)
            "ichi_pct": p.get("ichi_pct"),  # 1号の予測勝率→崩れ率badge用
            "exacta3": p["exacta3"],
            "exacta3_p": p.get("exacta3_p"),   # EV算出用
            "trio4": p.get("trio4"),           # 3連複4点(荒れ読みの頑健な器)
            "trio4_p": p.get("trio4_p"),
            "trifecta3": p.get("trifecta3"),   # 3連単3点(着順あり・大きい方)
            "trifecta3_p": p.get("trifecta3_p"),
            "trio_rank": p.get("trio_rank"),        # 全ランク(買い方くらべ日次のワイド系用)
            "trifecta_rank": p.get("trifecta_rank"),
            "deadline": p.get("deadline"),
            "exacta_ev": None,                 # 締切間際にcapture_evで埋める
            "logged_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "settled": False,
        })
        added += 1
        print(f"  記録 {p['venue']}{p['rno']}R 単勝{p['honmei']}号")
    _save(rows)
    print(f"\n{added}件を記録（台帳: {LEDGER}）")


def cmd_capture_ev(today: str):
    """未精算の本日ピックの2連単締切オッズを取得しEVを記録（overwrite=最新に収束）。

    締切間際に定期実行する想定。EV>2.0が検証済みの買い帯。settle前に締切値が入る。
    """
    from . import scraper, predict
    rows = _load()
    n = 0
    for r in rows:
        if r.get("settled") or r.get("date") != today or not r.get("exacta3_p"):
            continue
        lv = scraper.fetch_exacta_odds(r["date"], r["jcd"], r["rno"])
        ev = predict.exacta_ev(r.get("exacta3"), r.get("exacta3_p"), lv)
        if ev is not None:
            r["exacta_ev"] = round(ev, 3)
            n += 1
    _save(rows)
    print(f"{n}件のEVを更新（EV>{predict.EXACTA_EV_THRESHOLD}が買い帯）")


def cmd_capture_trifecta(today: str):
    """未精算の本日ピックの3連単締切オッズ(上位20点)を記録（overwrite=締切間際に収束）。

    狙い＝万舟側(3連単)の『締切表示 vs 確定払戻』乖離をライブ実測する布石。
    的中した万舟の deadline_odds×100 と settled払戻を後で比べれば、薄プール自己インパクト
    (=319%の"勝ちすぎ"がライブでどれだけ削れるか)の proxy that取れる。分析はN蓄積後。
    """
    from . import scraper
    rows = _load()
    n = 0
    for r in rows:
        if r.get("settled") or r.get("date") != today or not r.get("trifecta_rank"):
            continue
        od = scraper.fetch_trifecta_odds(r["date"], r["jcd"], r["rno"])
        if not od:
            continue
        top20 = (r.get("trifecta_rank") or [])[:20]
        r["trifecta_odds"] = {c: od[c] for c in top20 if c in od}   # 締切近辺odds(overwrite)
        n += 1
    _save(rows)
    print(f"{n}件の3連単締切オッズを記録（上位20点・万舟乖離の実測用）")


def cmd_settle(today: str):
    """確定済みレースを精算（単勝フラット100円。確定オッズ=払戻として実払戻計算）。"""
    rows = _load()
    n = 0
    for r in rows:
        if r.get("settled"):
            continue
        if r["date"] > today:    # 未来日は当然スキップ
            continue
        # 当日でも結果が出ていれば精算（未確定レースは fetch_result が None）
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
        # --- 2連単（上位3点）も精算 ---
        ex = scraper.fetch_exacta_payout(r["date"], r["jcd"], r["rno"])
        picks = r.get("exacta3", [])
        ex_hit = bool(ex and ex[0] in picks)
        r["exacta_result"] = ex[0] if ex else None
        r["exacta_points"] = len(picks)
        r["exacta_win"] = ex_hit
        r["exacta_return"] = ex[1] if ex_hit else 0     # 100円あたり払戻
        # --- 3連複（上位4点）も精算（着順不問セット。確定払戻） ---
        tr = scraper.fetch_trio_payout(r["date"], r["jcd"], r["rno"])
        tpicks = r.get("trio4", []) or []
        tr_hit = bool(tr and tr[0] in tpicks)
        r["trio_result"] = tr[0] if tr else None
        r["trio_points"] = len(tpicks)
        r["trio_win"] = tr_hit
        r["trio_return"] = tr[1] if tr_hit else 0       # 100円あたり払戻
        # --- 3連単（上位3点・着順あり）も精算（大きい方・別枠track） ---
        tf = scraper.fetch_trifecta_payout(r["date"], r["jcd"], r["rno"])
        fpicks = r.get("trifecta3", []) or []
        tf_hit = bool(tf and tf[0] in fpicks)
        r["trifecta_result"] = tf[0] if tf else None
        r["trifecta_points"] = len(fpicks)
        r["trifecta_win"] = tf_hit
        r["trifecta_return"] = tf[1] if tf_hit else 0
        # 万舟乖離の実測: 実的中組の締切オッズ(capture済なら)。deadline_odds×100 vs 確定tf[1] を後で比較
        r["trifecta_win_dl_odds"] = (r.get("trifecta_odds") or {}).get(tf[0]) if tf else None
        # 確定結果ブロブ（買い方くらべ日次＝styles.compute(pick,res)用。全ランクと合わせ全スタイル再計算可）
        r["res_full"] = {
            "winner": winner, "tansho_yen": r["tansho_return"],
            "exacta_combo": ex[0] if ex else None, "exacta_yen": (ex[1] if ex else None),
            "trio_combo": tr[0] if tr else None, "trio_yen": (tr[1] if tr else None),
            "trifecta_combo": tf[0] if tf else None, "trifecta_yen": (tf[1] if tf else None),
        }
        r["settled"] = True
        n += 1
    _save(rows)
    print(f"{n}件を精算")


# 単勝オッズ下限（クリーン年検証: 低オッズ単勝は控除率負け。1.5倍未満は見送りが有利）
ODDS_FLOOR = 1.5


def portfolio_stats(rows: list[dict]) -> dict:
    """精算済み記録から 単勝/2連単/合計 の集計を返す（1点100円）。"""
    settled = [r for r in rows if r.get("settled")]
    n = len(settled)
    t_stake = n * 100
    t_ret = sum(r.get("tansho_return", 0) for r in settled)
    t_hit = sum(1 for r in settled if r.get("tansho_win"))
    # 単勝オッズ1.5+のみ（低オッズ見送りルール適用時）
    hi = [r for r in settled if (r.get("final_odds") or 0) >= ODDS_FLOOR]
    hi_stake = len(hi) * 100
    hi_ret = sum(r.get("tansho_return", 0) for r in hi)
    hi_hit = sum(1 for r in hi if r.get("tansho_win"))
    e_stake = sum(r.get("exacta_points", 0) for r in settled) * 100
    e_ret = sum(r.get("exacta_return", 0) for r in settled)
    e_hit = sum(1 for r in settled if r.get("exacta_win"))
    # 3連複（上位4点）: 荒れ読みの頑健な器。確定払戻ベースの前向きtrack
    tr_settled = [r for r in settled if r.get("trio_points")]
    tr_stake = sum(r.get("trio_points", 0) for r in settled) * 100
    tr_ret = sum(r.get("trio_return", 0) for r in settled)
    tr_hit = sum(1 for r in settled if r.get("trio_win"))
    # 💎最強妙味: 本命勝率(win_pct)≥45 かつ 締切2連単EV>3.5 の2連単track（最濃断面の前向き実績）
    from .predict import STRONGEST_P0, STRONGEST_EV
    strong = [r for r in settled if (r.get("win_pct") or 0) >= STRONGEST_P0 * 100
              and (r.get("exacta_ev") or 0) > STRONGEST_EV and r.get("exacta_points")]
    s_stake = sum(r.get("exacta_points", 0) for r in strong) * 100
    s_ret = sum(r.get("exacta_return", 0) for r in strong)
    s_hit = sum(1 for r in strong if r.get("exacta_win"))
    # 3連単（上位3点・着順あり）: 大きい方・fat-tail。別枠track
    tf_settled = [r for r in settled if r.get("trifecta_points")]
    tf_stake = sum(r.get("trifecta_points", 0) for r in settled) * 100
    tf_ret = sum(r.get("trifecta_return", 0) for r in settled)
    tf_hit = sum(1 for r in settled if r.get("trifecta_win"))
    # 推奨: 単勝<1.5(チャラレース)を丸ごと見送った場合の 単勝+2連単（検証で最良）
    reco = [r for r in settled if (r.get("final_odds") or 0) >= ODDS_FLOOR]
    rc_stake = len(reco) * 100 + sum(r.get("exacta_points", 0) for r in reco) * 100
    rc_ret = sum(r.get("tansho_return", 0) + r.get("exacta_return", 0) for r in reco)
    # 2連単EV>2.0フィルタ（検証: 締切EV>2.0帯が堅く300-500%・fat-tail生存）
    from .predict import EXACTA_EV_THRESHOLD
    evf = [r for r in settled if (r.get("exacta_ev") or 0) >= EXACTA_EV_THRESHOLD]
    evf_stake = sum(r.get("exacta_points", 0) for r in evf) * 100
    evf_ret = sum(r.get("exacta_return", 0) for r in evf)
    evf_hit = sum(1 for r in evf if r.get("exacta_win"))
    captured = [r for r in settled if r.get("exacta_ev") is not None]
    return {
        "races": n,
        "tansho": {"stake": t_stake, "ret": t_ret, "hit": t_hit, "pl": t_ret - t_stake},
        "tansho_hi": {"stake": hi_stake, "ret": hi_ret, "hit": hi_hit,
                      "pl": hi_ret - hi_stake, "races": len(hi)},
        "exacta": {"stake": e_stake, "ret": e_ret, "hit": e_hit, "pl": e_ret - e_stake},
        "trio": {"stake": tr_stake, "ret": tr_ret, "hit": tr_hit,
                 "pl": tr_ret - tr_stake, "races": len(tr_settled)},
        "trifecta": {"stake": tf_stake, "ret": tf_ret, "hit": tf_hit,
                     "pl": tf_ret - tf_stake, "races": len(tf_settled)},
        "total": {"stake": t_stake + e_stake, "ret": t_ret + e_ret,
                  "pl": (t_ret + e_ret) - (t_stake + e_stake)},
        "reco": {"races": len(reco), "skipped": n - len(reco),
                 "stake": rc_stake, "ret": rc_ret, "pl": rc_ret - rc_stake},
        "exacta_ev": {"races": len(evf), "captured": len(captured),
                      "stake": evf_stake, "ret": evf_ret, "hit": evf_hit,
                      "pl": evf_ret - evf_stake},
        "strongest": {"races": len(strong), "stake": s_stake, "ret": s_ret,
                      "hit": s_hit, "pl": s_ret - s_stake},
    }


def cmd_report():
    rows = [r for r in _load() if r.get("settled")]
    if not rows:
        print("精算済みの記録がまだありません（log→（結果後）settle）。")
        return
    s = portfolio_stats(rows)
    n = s["races"]

    def line(name, d, hit=None):
        roi = d["ret"] / d["stake"] if d["stake"] else 0
        h = f" 的中{hit}" if hit is not None else ""
        print(f"  {name:<8} 賭金{d['stake']:>6,}円 回収{roi*100:>5.1f}% 収支{d['pl']:>+7,}円{h}")

    print(f"=== 実トラック（{n}レース・1点100円）===")
    line("単勝", s["tansho"], s["tansho"]["hit"])
    line("2連単3点", s["exacta"], s["exacta"]["hit"])
    if s["trio"]["races"]:
        line("3連複4点", s["trio"], s["trio"]["hit"])
    if s["trifecta"]["races"]:
        line("3連単3点", s["trifecta"], s["trifecta"]["hit"])
    line("合計", s["total"])
    rc = s["reco"]
    line("推奨(単+2連)", rc)
    ev = s["exacta_ev"]
    if ev["captured"]:
        line(f"2連単EV>2.0", ev, ev["hit"])
        print(f"    （EV捕捉 {ev['captured']}レース中 EV>2.0で買い {ev['races']}レース）")
    sg = s["strongest"]
    if sg["races"]:
        line("💎最強妙味", sg, sg["hit"])
        print(f"    （本命勝率≥45%×締切EV>3.5＝最濃断面。backtest 592%・全年・頑健）")
    print(f"\n  ※「推奨」= 単勝1.5倍未満のレースを丸ごと見送った単勝+2連単（検証で最良）。"
          f"見送り {rc['skipped']}レース。")
    print(f"  ※「2連単EV>2.0」= 締切EVフィルタの前向きtrack（机上で堅く300-500%・要ライブ確認）。")
    print(f"  ※前向き記録だけが本物。1日でなく数十〜百本の平均で判断。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log"); lg.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    sub.add_parser("settle")
    sub.add_parser("report")
    sub.add_parser("capture_ev")        # 締切間際に定期実行=2連単EVを台帳に記録
    sub.add_parser("capture_trifecta")  # 締切間際=3連単締切オッズ(万舟乖離の実測用)
    args = ap.parse_args()
    today = dt.date.today().strftime("%Y%m%d")
    if args.cmd == "log":
        cmd_log(args.date)
    elif args.cmd == "settle":
        cmd_settle(today)
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "capture_ev":
        cmd_capture_ev(today)
    elif args.cmd == "capture_trifecta":
        cmd_capture_trifecta(today)


if __name__ == "__main__":
    main()
