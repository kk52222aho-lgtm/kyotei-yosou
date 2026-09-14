"""②の採点を **DBの確定払戻** でやる版。判定ロジックは `odds_probe_score` から素で借りる。

=== なぜ別ファイルにしたか ===
`odds_probe_score.settle` は1レースずつ **実フェッチ**する。やが `scraper._get` は
**成功時に待機を入れとらん**(SLEEP_SEC は再試行時だけ)。今回の母数は約6,000レースで、
それを無制限に boatrace.jp へ投げることになる。**同じホストに、今日復旧させたばかりの
前向き収集ループが当たっとる。**ブロックされたら締切前オッズが永久欠損する
(9/05-9/08 で3日分すでに失っとる)。

→ **判定は1回きりやが、収集は毎日続く。**収集を危険に晒してまで急ぐ理由が無い。

`odds_probe_score.py` は**1バイトも触っとらん**。発火判定 `fired`、CI `cluster_ci`、
判定 `judge`、閾値 `N_TARGET`、窓 `LATE` を**そのまま import** する。
差し替えたんは**払戻の取得先だけ**や(ネット → 手元の `payouts` 表)。

`payouts` は公式Kファイルを `official.parse_k` が読んだもんで、
サイトが表示しとる確定払戻と同一のはず。**「はず」で済まさんために、
無作為20レースを実フェッチして突き合わせてから採点する。**

usage: python -u -m src.odds_probe_score_db
"""
from __future__ import annotations

import random

import pandas as pd

from . import storage, scraper
from .odds_probe_score import fired, cluster_ci, judge, N_TARGET, LATE


def payouts_map(conn) -> dict:
    d = pd.read_sql_query(
        "SELECT date, jcd, rno, tansho_lane, tansho_yen FROM payouts "
        "WHERE tansho_lane IS NOT NULL", conn)
    d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    return {(r.date, r.jcd, int(r.rno)): (int(r.tansho_lane), float(r.tansho_yen))
            for r in d.itertuples()}


def verify(keys, pm, n=20, seed=20260908):
    """DBの払戻がサイトの確定払戻と一致するかを、無作為nレースで突き合わせる。"""
    random.seed(seed)
    sample = random.sample(sorted(keys), min(n, len(keys)))
    ok = bad = miss = 0
    for (date, jcd, rno) in sample:
        res = scraper.fetch_result_full(date, jcd, rno)
        if not res or not res.get("winner"):
            miss += 1
            continue
        w, y = pm[(date, jcd, rno)]
        if int(res["winner"]) == w and abs(float(res.get("tansho_yen") or 0) - y) < 1:
            ok += 1
        else:
            bad += 1
            print(f"    不一致 {date} {jcd} {rno}R: "
                  f"site={res['winner']}/{res.get('tansho_yen')} db={w}/{y}")
    print(f"  突合 {len(sample)}レース: 一致 {ok} / 不一致 {bad} / 取得不可 {miss}")
    return bad == 0 and ok >= 1


def main():
    conn = storage.connect()
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM odds_timeseries ORDER BY date")]
    bets = []
    for d in dates:
        bets += fired(conn, d)
    pm = payouts_map(conn)
    conn.close()
    print(f"収集日 {len(dates)}日 ({dates[0]}-{dates[-1]})  発火 {len(bets):,}件")

    keys = {(b["date"], b["jcd"], b["rno"]) for b in bets} & set(pm)
    print(f"払戻が手元にあるレース {len(keys):,} / 発火レース "
          f"{len({(b['date'], b['jcd'], b['rno']) for b in bets}):,}")

    print("\n=== 払戻の出どころを突き合わせる(採点の前) ===")
    if not verify(keys, pm):
        print("  ✗ 一致せんかった。DBを採点に使わん。中止。")
        return
    print("  ✓ 一致。DBの払戻で採点する")

    settled = []
    for b in bets:
        k = (b["date"], b["jcd"], b["rno"])
        if k not in pm:
            continue
        w, y = pm[k]
        won = int(w == b["target"])
        settled.append({**b, "won": won, "ret": y if won else 0.0})

    print("\n=== 前向き成績(事前登録 = src/odds_probe.py の docstring) ===")
    print(f"  {'ルール':<6}{'N':>7}{'的中':>7}{'的中率':>8}{'ROI':>8}"
          f"{'CI(regクラスタ)':>20}   判定")
    for rule in ["R1", "R2", "R3", "R4"]:
        rows = [s for s in settled if s["rule"] == rule]
        n = len(rows)
        if n == 0:
            print(f"  {rule:<6}      0  —  発火・確定 0件")
            continue
        hit = sum(r["won"] for r in rows)
        roi = sum(r["ret"] for r in rows) / (n * 100) * 100
        lo, hi = cluster_ci(rows)
        print(f"  {rule:<6}{n:>7,}{hit:>7,}{hit / n * 100:>7.1f}%{roi:>7.0f}%"
              f"   [{lo:>6.0f},{hi:>6.0f}]   {judge(n, roi, lo, hi)}")
    print(f"\n※ 合格=N≥{N_TARGET}&CI下限≥100 / kill=N≥100&CI上限<100 or "
          f"N≥{N_TARGET}&ROI<90 / 継続=CIが100跨ぎ")
    print(f"  late窓 {LATE}(T-1優先→T-3)・単勝・確定払戻。閾値を動かす=新規登録でN振り直し。")


if __name__ == "__main__":
    main()
