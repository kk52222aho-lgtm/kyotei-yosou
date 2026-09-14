"""オリジナル展示 (一周/まわり足/直線) の予測力・増分エッジ一次検証。

問い: BOATCAST集約のオリジナル展示数値は、着順に対して
  (1) そもそも予測力があるか (レース内順位 vs 着)
  (2) 既知情報を超える増分があるか
      - コース(lane) を超えるか
      - 既存 entries.tenji_time (展示タイム) を超えるか
を測る。場ごとにラベル/スケールが違うため、指標は全て「レース内順位」で扱う
(桐生の半周ラップは lap 相当として lap に合流)。

使い方: python test_oriten_edge.py --start 20260620 --end 20260629
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

import storage


def load(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(storage.DB_PATH)
    o = pd.read_sql(
        "SELECT date,jcd,rno,lane,lap,halflap,mawariashi,straight "
        "FROM oriten WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    e = pd.read_sql(
        "SELECT date,jcd,rno,lane,finish,tenji_time FROM entries "
        "WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    conn.close()
    df = o.merge(e, on=["date", "jcd", "rno", "lane"], how="inner")
    # 一周相当: lap 無し(桐生)は halflap で代用。順位化するので統合可。
    df["lap_metric"] = df["lap"].fillna(df["halflap"])
    df["is_win"] = (df["finish"] == 1).astype(int)
    df["is_top2"] = (df["finish"].isin([1, 2])).astype(int)
    return df


def rank_in_race(df: pd.DataFrame, col: str) -> pd.Series:
    # 速い(小さい)ほど 1位
    return df.groupby(["date", "jcd", "rno"])[col].rank(method="min", ascending=True)


def ev_backtest(start: str, end: str) -> None:
    """実配当(payouts.tansho_yen)に対する単勝ROIをブートストラップCI付きで検定。

    公開信号は市場に織り込まれるのが常(他ポートの結論)。予測力があっても
    ROIが1.0を有意に超えなければ回収エッジは無い。点推定でなく95%CIで判定する。
    """
    conn = sqlite3.connect(storage.DB_PATH)
    o = pd.read_sql(
        "SELECT date,jcd,rno,lane,lap,halflap,mawariashi,straight "
        "FROM oriten WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    e = pd.read_sql(
        "SELECT date,jcd,rno,lane,finish,tenji_time FROM entries "
        "WHERE date BETWEEN ? AND ?", conn, params=(start, end))
    p = pd.read_sql(
        "SELECT date,jcd,rno,tansho_lane,tansho_yen FROM payouts "
        "WHERE date BETWEEN ? AND ? AND tansho_yen IS NOT NULL",
        conn, params=(start, end))
    conn.close()
    df = o.merge(e, on=["date", "jcd", "rno", "lane"])
    df["lap_metric"] = df["lap"].fillna(df["halflap"])

    print(f"\n=== 単勝EVバックテスト (実配当) payouts重なり {p.groupby(['date','jcd','rno']).ngroups}レース ===")
    rng = np.random.default_rng(0)

    def roi_ci(pick: pd.DataFrame, label: str):
        m = pick.merge(p, on=["date", "jcd", "rno"])
        if len(m) == 0:
            print(f"  [{label}] 重なりなし")
            return
        ret = np.where(m["pl"] == m["tansho_lane"], m["tansho_yen"], 0.0).astype(float)
        n = len(ret)
        roi = ret.sum() / (100 * n)
        # ブートストラップでROIの95%CI
        boot = [rng.choice(ret, n, replace=True).sum() / (100 * n) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hitr = (ret > 0).mean()
        flag = "＋エッジ疑" if lo > 1.0 else ("×市場効率" if hi < 1.0 else "±有意差なし")
        print(f"  [{label:14s}] n={n} 的中{hitr:.1%} ROI={roi:.3f} "
              f"95%CI[{lo:.3f},{hi:.3f}] {flag}")

    for col in ["lap_metric", "mawariashi", "straight", "tenji_time"]:
        df["rk"] = df.groupby(["date", "jcd", "rno"])[col].rank(method="first", ascending=True)
        pick = df[df["rk"] == 1][["date", "jcd", "rno", "lane"]].rename(columns={"lane": "pl"})
        roi_ci(pick, col + "1位")
    # 参照: 1号艇ベタ (市場効率のベンチ)
    pick = df[df["lane"] == 1][["date", "jcd", "rno", "lane"]].rename(columns={"lane": "pl"})
    roi_ci(pick, "1号艇ベタ")
    print("  判定: CI下限>1.0で回収エッジ疑い / CI上限<1.0で市場効率(織込済) / またぐ=有意差なし")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--ev", action="store_true", help="単勝EVバックテストも実行")
    args = ap.parse_args()

    df = load(args.start, args.end)
    nrace = df.groupby(["date", "jcd", "rno"]).ngroups
    print(f"結合: {len(df)}艇 / {nrace}レース / "
          f"{df['jcd'].nunique()}場 / {df['date'].nunique()}日")
    if nrace == 0:
        print("データ無し。収集が済んでいるか確認。")
        return

    metrics = ["lap_metric", "mawariashi", "straight", "tenji_time"]
    for m in metrics:
        df[m + "_rk"] = rank_in_race(df, m)

    base = df["is_win"].mean()
    print(f"\nベース1着率(全艇) = {base:.3f}")
    print("── 各指標 rank1 艇の1着率 (予測力の一次読み) ──")
    for m in metrics:
        sub = df[df[m + "_rk"] == 1]
        cov = sub[m].notna().mean()
        print(f"  {m:12s} rank1→1着率 {sub['is_win'].mean():.3f} "
              f"top2率 {sub['is_top2'].mean():.3f} (n={len(sub)})")

    # rank ごとの平均着順(単調なら予測力あり)
    print("\n── lap_metric rank 別 平均着順 (単調に増えれば信号) ──")
    g = df.dropna(subset=["lap_metric"]).groupby("lap_metric_rk")["finish"].agg(["mean", "count"])
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))

    # 増分1: コース(lane)を超えるか — 1号艇を lap_rank で分割
    print("\n── 増分検定A: 1号艇の1着率を lap_metric rank で分割 ──")
    inn = df[df["lane"] == 1]
    b = inn["is_win"].mean()
    print(f"  1号艇 全体1着率 = {b:.3f} (n={len(inn)})")
    hi = inn[inn["lap_metric_rk"] <= 2]["is_win"].mean()
    lo = inn[inn["lap_metric_rk"] >= 4]["is_win"].mean()
    print(f"    うち lap上位(rk<=2) = {hi:.3f} / lap下位(rk>=4) = {lo:.3f} → 差 {hi-lo:+.3f}")

    # 増分2: 既存 tenji_time を超えるか — lap_rank と tenji_rank の一致/不一致で
    print("\n── 増分検定B: tenji_time順位を既知として、lap順位が追加情報か ──")
    d2 = df.dropna(subset=["lap_metric", "tenji_time"]).copy()
    # tenji上位2かつ lap上位2 vs tenji上位2だが lap下位
    tt_top = d2[d2["tenji_time_rk"] <= 2]
    both = tt_top[tt_top["lap_metric_rk"] <= 2]["is_win"].mean()
    only_tt = tt_top[tt_top["lap_metric_rk"] >= 4]["is_win"].mean()
    print(f"  tenji上位2 かつ lap上位2 → 1着率 {both:.3f}")
    print(f"  tenji上位2 だが lap下位(>=4) → 1着率 {only_tt:.3f}")
    print(f"  → lap の追加寄与 (同じtenji上位でも) 差 {both-only_tt:+.3f}")

    # 相関(Spearman): 各指標rank と finish
    print("\n── Spearman相関 (指標rank vs 着順、正=速い艇ほど上位着) ──")
    for m in metrics:
        s = df.dropna(subset=[m])
        rho = s[m + "_rk"].corr(s["finish"], method="spearman")
        print(f"  {m:12s} rho = {rho:+.3f}")

    if args.ev:
        ev_backtest(args.start, args.end)


if __name__ == "__main__":
    main()
