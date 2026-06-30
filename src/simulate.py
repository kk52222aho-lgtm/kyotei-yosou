"""バンクロール・シミュレーション。

検証済みの「勝てる形」を時系列で実際に賭け続け、資金曲線・最大ドローダウン・
賭け頻度・連敗を出す。回収率が100%超でも、賭け機会が少なく/連敗が深ければ
実運用に耐えない。そこを確かめる。

戦略:
  - 単勝コントラリアン: モデル本命≠1号艇のとき本命を単勝（1点/レース）
  - ポートフォリオ: 上記＋2連単・確率上位3点（計4点/レース）

例:
  python -m src.simulate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .strategy_search import build_records, exacta_probs, _fit_predict


def _equity_stats(pl_seq: list[int]) -> dict:
    """損益系列から 最大DD・最長連敗・最終資金 を計算。"""
    eq = np.cumsum(pl_seq)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([0])
    dd = (eq - peak)
    max_dd = int(dd.min()) if len(dd) else 0
    # 連敗(レース単位の負け)の最長
    streak = worst = 0
    for x in pl_seq:
        if x < 0:
            streak += 1; worst = max(worst, streak)
        else:
            streak = 0
    return {"最終損益": int(eq[-1]) if len(eq) else 0,
            "最大DD": max_dd, "最長連敗": worst}


def simulate(recs, stake=100, portfolio=False):
    """recs を時系列順に賭ける。返り値: レース毎損益リスト + 集計。"""
    pl = []
    bets = hits = ret = 0
    for r in recs:
        if r["top1"] == 1:           # 本命=1号艇は見送り
            continue
        race_stake = 0
        race_ret = 0
        # 単勝（本命1点）
        if r["tansho_lane"] is not None and not pd.isna(r["tansho_lane"]):
            race_stake += stake
            if int(r["tansho_lane"]) == r["top1"]:
                race_ret += stake * r["tansho_yen"] / 100
        # ポートフォリオ: 2連単 上位3点
        if portfolio and r["exacta_combo"] is not None and not pd.isna(r["exacta_yen"]):
            picks = r["ex_rank"][:3]
            race_stake += stake * len(picks)
            if r["exacta_combo"] in picks:
                race_ret += stake * r["exacta_yen"] / 100
        if race_stake == 0:
            continue
        bets += race_stake // stake
        ret += race_ret
        if race_ret > 0:
            hits += 1
        pl.append(int(race_ret - race_stake))
    stats = _equity_stats(pl)
    races = len(pl)
    stats.update({
        "レース": races, "賭点": bets,
        "回収率": ret / (bets * stake) if bets else 0,
        "的中率": hits / races if races else 0,
    })
    return pl, stats


def main():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.exacta_combo, p.exacta_yen,
               p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()

    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    tr = df["date"].to_numpy() < cut
    te = ~tr
    test = _fit_predict(df, tr, te).sort_values(["date", "jcd", "rno"])
    recs = build_records(test)  # 時系列順を保持

    total_races = len({(r,) for r in range(len(recs))})  # = len(recs)
    print(f"テスト期間 {cut}以降 / 全{len(recs)}レース（うち賭け対象=本命≠1号艇）\n")

    pd.set_option("display.unicode.east_asian_width", True)
    for label, pf in [("単勝コントラリアン", False), ("ポートフォリオ(単勝+2連単3点)", True)]:
        pl, st = simulate(recs, portfolio=pf)
        bet_rate = st["レース"] / len(recs) if recs else 0
        print(f"■ {label}")
        print(f"  賭けたレース: {st['レース']} / {len(recs)}  (賭け頻度 {bet_rate:.1%})")
        print(f"  回収率: {st['回収率']:.1%}   的中率: {st['的中率']:.1%}")
        print(f"  最終損益: {st['最終損益']:+,}円  (賭金総額 {st['賭点']*100:,}円)")
        print(f"  最大ドローダウン: {st['最大DD']:,}円   最長連敗: {st['最長連敗']}レース\n")

    # 月別の資金推移（単勝コントラリアン）
    print("■ 単勝コントラリアンの月別収支")
    test2 = test.copy()
    # レース→月、賭け対象のみ
    rows = []
    for r, (_, g) in zip(recs, test.groupby(["date", "jcd", "rno"])):
        if r["top1"] == 1 or pd.isna(r["tansho_lane"]):
            continue
        ym = g.iloc[0]["date"][:6]
        ret = r["tansho_yen"] if int(r["tansho_lane"]) == r["top1"] else 0
        rows.append((ym, ret - 100))
    M = pd.DataFrame(rows, columns=["月", "損益"])
    if len(M):
        agg = M.groupby("月").agg(レース=("損益", "size"), 収支=("損益", "sum"))
        agg["回収率"] = (agg["収支"] + agg["レース"] * 100) / (agg["レース"] * 100)
        agg["回収率"] = (agg["回収率"] * 100).round(0).astype(int).astype(str) + "%"
        agg["収支"] = agg["収支"].map(lambda v: f"{int(v):+,}")
        print(agg.to_string())
    print("\n※flat100円/点。少額分散前提。控除率25%下で100%超なら理論上プラス。")


if __name__ == "__main__":
    main()
