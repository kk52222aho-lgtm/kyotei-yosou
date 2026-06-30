"""3連単 期待値(EV)バックテスト。

単勝フラットは控除率の壁を越えられなかった（docs/FINDINGS.md）。
3連単は高配当＝市場が穴目で非効率になりやすいので、EVベットで100%超を狙えるか検証する。

手順:
  1. テスト期間より前のデータでモデル学習
  2. テスト期間のレースをサンプリング
  3. 各レース: モデル勝率 → Harville で120通りの確率 → odds3t(確定オッズ)取得 → EV
  4. 戦略別に買い目を決め、実際の的中(payouts)と突合して回収率を集計

過去レースの odds3t は確定オッズを保持。スクレイプ量を抑えるため --sample で件数制限。

例:
  python -m src.backtest_trifecta --start 20260601 --end 20260618 --sample 300
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import scraper, storage
from .features import FEATURES, build_frame
from .predict import harville_trifecta
from .train import _make_estimator


def _train_model(before_date: str):
    conn = storage.connect()
    df = pd.read_sql_query(
        "SELECT * FROM entries WHERE win IS NOT NULL AND date < ?", conn, params=(before_date,))
    conn.close()
    X = build_frame(df)[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    m.fit(X, y)
    return m


def _test_races(start: str, end: str):
    """払戻のあるテストレース一覧 [(date,jcd,rno,trifecta_combo,trifecta_yen), ...]"""
    conn = storage.connect()
    rows = conn.execute("""
        SELECT date,jcd,rno,trifecta_combo,trifecta_yen FROM payouts
        WHERE date>=? AND date<=? AND trifecta_yen IS NOT NULL
        ORDER BY date,jcd,rno
    """, (start, end)).fetchall()
    conn.close()
    return rows


def _race_entries(date, jcd, rno):
    conn = storage.connect()
    df = pd.read_sql_query(
        "SELECT * FROM entries WHERE date=? AND jcd=? AND rno=? ORDER BY lane",
        conn, params=(date, jcd, rno))
    conn.close()
    return df


class Strat:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn
        self.bets = self.hits = self.ret = self.races = 0

    def play(self, ranked_ev, win_combo, win_yen):
        picks = self.fn(ranked_ev)   # 買う買い目の集合
        if not picks:
            return
        self.races += 1
        self.bets += len(picks)
        if win_combo in picks:
            self.hits += 1
            self.ret += win_yen

    def row(self):
        stake = self.bets * 100
        return {"戦略": self.name, "レース": self.races, "賭点": self.bets,
                "的中率": f"{self.hits/self.races:.1%}" if self.races else "-",
                "回収率": f"{self.ret/stake:.1%}" if stake else "-",
                "収支": f"{self.ret-stake:+,}円"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260601")
    ap.add_argument("--end", default="20260618")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--ev", type=float, default=1.2, help="EV閾値")
    args = ap.parse_args()

    print(f"学習: {args.start}より前 / テスト: {args.start}〜{args.end}")
    model = _train_model(args.start)
    bundle = {"model": model, "features": FEATURES}

    races = _test_races(args.start, args.end)
    if len(races) > args.sample:
        idx = np.linspace(0, len(races) - 1, args.sample).astype(int)
        races = [races[i] for i in idx]
    print(f"対象 {len(races)} レース（オッズ取得中… 1件約1秒）")

    # 戦略: EV閾値以上を買う / EV上位N点 / Harville確率上位 / 本命形1-2-3固定
    EV = args.ev
    strategies = [
        Strat(f"EV>{EV} 全買い", lambda r: {x["combo"] for x in r if x["ev"] > EV}),
        Strat("EV上位3点", lambda r: {x["combo"] for x in sorted(r, key=lambda d:-d["ev"])[:3]}),
        Strat("確率上位3点", lambda r: {x["combo"] for x in sorted(r, key=lambda d:-d["prob"])[:3]}),
        Strat("確率1位の1点", lambda r: {max(r, key=lambda d: d["prob"])["combo"]} if r else set()),
        Strat("1-2-3固定(ベース)", lambda r: {"1-2-3"}),
    ]

    used = 0
    for (date, jcd, rno, win_combo, win_yen) in races:
        odds = scraper.fetch_trifecta_odds(date, jcd, rno)
        if not odds:
            continue
        df = _race_entries(date, jcd, rno)
        if len(df) < 6:
            continue
        df = df.copy()
        df["jcd"] = jcd
        X = build_frame(df)[FEATURES].to_numpy(dtype=float)
        raw = model.predict_proba(X)[:, 1]
        win_prob = {int(l): float(p) for l, p in zip(df["lane"], raw)}
        probs = harville_trifecta(win_prob)
        ranked = [{"combo": c, "prob": p, "odds": odds[c], "ev": p * odds[c]}
                  for c, p in probs.items() if c in odds]
        for s in strategies:
            s.play(ranked, win_combo, win_yen)
        used += 1

    print(f"\n有効 {used} レース（3連単オッズ取得成功）\n")
    rep = pd.DataFrame([s.row() for s in strategies])
    pd.set_option("display.unicode.east_asian_width", True)
    print(rep.to_string(index=False))
    print("\n※回収率100%超で理論上プラス。3連単の控除率は約25%。"
          "EVはHarville確率×確定オッズ（穴目はEV過大に出やすい点に注意）。")


if __name__ == "__main__":
    main()
