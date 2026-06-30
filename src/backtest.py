"""回収率バックテスト。

学習期間でモデルを学習し、**期間外のテスト期間**で「実際に賭けたら儲かるか」を検証する。
払戻金は K ファイル由来（100円賭けあたりの金額）。

評価指標:
  的中率 = 当たった賭け数 / 賭けた数
  回収率(ROI) = 払戻合計 / 賭け金合計   （100% を超えれば理論上プラス）

例:
  python -m src.backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator


def load() -> pd.DataFrame:
    conn = storage.connect()
    # 払戻データのあるレースのみ（公式データ期間）。INNER JOIN で欠損レースを除外。
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.exacta_combo, p.exacta_yen,
               p.quinella_combo, p.quinella_yen, p.tansho_lane, p.tansho_yen
        FROM entries e
        JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    return df


def _has(v) -> bool:
    """払戻値が有効か（None / NaN を除外）。"""
    return v is not None and not (isinstance(v, float) and pd.isna(v))


class Strategy:
    """1レースにつき (賭け点数, 払戻円) を返す賭け方。100円/点。"""
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
        self.bets = 0       # 賭けた点数(=100円単位)
        self.hits = 0       # 当たった点数
        self.ret = 0        # 払戻合計(円)
        self.races = 0      # 参加レース数

    def play(self, ranked, pay):
        res = self.fn(ranked, pay)
        if res is None:
            return
        n_points, payout = res
        if n_points <= 0:
            return
        self.races += 1
        self.bets += n_points
        self.ret += payout
        if payout > 0:
            self.hits += 1

    def report(self) -> dict:
        stake = self.bets * 100
        return {
            "戦略": self.name,
            "レース": self.races,
            "賭点": self.bets,
            "的中率": (self.hits / self.races) if self.races else 0,
            "回収率": (self.ret / stake) if stake else 0,
            "収支": self.ret - stake,
        }


# --- 賭け方の定義。ranked = P(win)降順の艇番リスト, pay = 払戻 dict ---

def s_tansho_honmei(ranked, pay):
    if not _has(pay.get("tansho_yen")):
        return None
    bet = ranked[0]
    win = pay["tansho_yen"] if pay["tansho_lane"] == bet else 0
    return (1, win)


def s_tansho_lane1(ranked, pay):
    if not _has(pay.get("tansho_yen")):
        return None
    win = pay["tansho_yen"] if pay["tansho_lane"] == 1 else 0
    return (1, win)


def s_exacta_honmei(ranked, pay):
    if not _has(pay.get("exacta_yen")) or len(ranked) < 2:
        return None
    combo = f"{ranked[0]}-{ranked[1]}"
    win = pay["exacta_yen"] if pay["exacta_combo"] == combo else 0
    return (1, win)


def s_exacta_nagashi(ranked, pay):
    """2連単 本命1着固定 → 相手 top2,3 の2点"""
    if not _has(pay.get("exacta_yen")) or len(ranked) < 3:
        return None
    combos = [f"{ranked[0]}-{ranked[1]}", f"{ranked[0]}-{ranked[2]}"]
    win = pay["exacta_yen"] if pay["exacta_combo"] in combos else 0
    return (2, win)


def s_trifecta_honmei(ranked, pay):
    if not _has(pay.get("trifecta_yen")) or len(ranked) < 3:
        return None
    combo = f"{ranked[0]}-{ranked[1]}-{ranked[2]}"
    win = pay["trifecta_yen"] if pay["trifecta_combo"] == combo else 0
    return (1, win)


def s_trifecta_formation(ranked, pay):
    """3連単 1着=本命固定, 2着=top2,3, 3着=top2,3,4 (重複除く=4点)"""
    if not _has(pay.get("trifecta_yen")) or len(ranked) < 4:
        return None
    a = ranked[0]
    combos = set()
    for b in (ranked[1], ranked[2]):
        for c in (ranked[1], ranked[2], ranked[3]):
            if len({a, b, c}) == 3:
                combos.add(f"{a}-{b}-{c}")
    win = pay["trifecta_yen"] if pay["trifecta_combo"] in combos else 0
    return (len(combos), win)


def make_value_tansho(threshold: float):
    """単勝・本命だが 期待値(P×払戻/100) が閾値超のレースだけ賭ける。
    払戻を実オッズの近似として使用（締切直前オッズ≈払戻）。"""
    def fn(ranked, pay):
        if not _has(pay.get("tansho_yen")):
            return None
        ev = ranked.p0 * (pay["tansho_yen"] / 100.0)
        if ev < threshold:
            return (0, 0)  # 見送り
        bet = ranked[0]
        win = pay["tansho_yen"] if pay["tansho_lane"] == bet else 0
        return (1, win)
    return fn


class Ranked(list):
    """P(win)降順の艇番リスト + 本命のP(win) p0 を保持。"""
    p0 = 0.0


def main():
    df = load()
    dates = np.sort(df["date"].unique())
    if len(dates) < 5:
        print("データ不足"); return
    cut = dates[int(len(dates) * 0.8)]
    tr = df["date"].to_numpy() < cut
    te = ~tr

    X = build_frame(df)[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    model = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    model.fit(X[tr], y[tr])
    proba = model.predict_proba(X[te])[:, 1]

    test = df[te].copy()
    test["p"] = proba

    strategies = [
        Strategy("単勝・本命", s_tansho_honmei),
        Strategy("単勝・1号艇(ベース)", s_tansho_lane1),
        Strategy("単勝・本命(期待値>1.1のみ)", make_value_tansho(1.1)),
        Strategy("2連単・本命", s_exacta_honmei),
        Strategy("2連単・本命1着-相手2点", s_exacta_nagashi),
        Strategy("3連単・本命", s_trifecta_honmei),
        Strategy("3連単・本命フォーメーション", s_trifecta_formation),
    ]

    for _, g in test.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        ranked = Ranked(int(x) for x in g["lane"])
        ranked.p0 = float(g["p"].iloc[0])
        pay = g.iloc[0].to_dict()  # 払戻はレース共通
        for s in strategies:
            s.play(ranked, pay)

    rep = pd.DataFrame([s.report() for s in strategies])
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\nテスト期間: {cut}以降  / 全 {test.groupby(['date','jcd','rno']).ngroups} レース\n")
    rep["的中率"] = (rep["的中率"] * 100).round(1).astype(str) + "%"
    rep["回収率"] = (rep["回収率"] * 100).round(1).astype(str) + "%"
    rep["収支"] = rep["収支"].map(lambda v: f"{v:+,}円")
    print(rep.to_string(index=False))
    print("\n※払戻は100円賭けあたり。回収率100%超で理論上プラス。"
          "期待値フィルタは払戻を実オッズの近似として使用。")


if __name__ == "__main__":
    main()
