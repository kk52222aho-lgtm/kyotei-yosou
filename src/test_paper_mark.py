"""プロの予想屋(宮島の紙の記者)は、盤より上か。

**これは探索やない。**比べるんは**先に固定した3つの手**だけで、
特徴を選んだり閾値を振ったりせん。せやから holdout の多重性を食わん
(念のため train/holdout も別々に出す)。

    ① 1号艇固定      … 予測が要らん手
    ② モデル本命      … 盤(既存14特徴の OOF 予測)
    ③ **記者の◎**    … 予想屋

測るんは2つ:

    本命的中率  … 店頭に出す数字([[feedback_sell_the_win]])
    単勝ROI     … 金の数字。到達率 = (ROI − 払戻率)/(1 − 払戻率)
                  → [[insight_required_vs_observed_error]]

🚨 **「6艇フラット全買いの回収率」を払戻率と呼んだらあかん。**
   最初そう書いて 68% が出たが、あれは払戻率やのうて
   「穴が過剰人気やから均等買いは払戻率より下に出る」いう別の量やった。
   払戻率は odds_timeseries の 1/Σ(1/オッズ) で実測 = 全24場6,568レースで中央 73.57%。

到達率 = (ROI − 払戻率)/(1 − 払戻率)。0%=無作為、100%=損益分岐。
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd

# 単勝の払戻率。odds_timeseries の 1/sum(1/odds) を全24場6,568レースで実測して
# 中央 73.57%(表示オッズの切り捨てで低めに出る)。公称の控除25%と整合するんで 0.75 を使う。
KAESHI = 0.75


def load(db: str, jcd: str = "17") -> pd.DataFrame:
    con = sqlite3.connect(db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    ent = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, finish FROM entries WHERE jcd=?", con, params=(jcd,))
    pay = pd.read_sql_query(
        "SELECT date, jcd, rno, tansho_lane, tansho_yen FROM payouts WHERE jcd=?",
        con, params=(jcd,))
    mk = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, mark FROM paper_mark WHERE jcd=?", con, params=(jcd,))
    con.close()
    for d in (ent, pay, mk):
        d["jcd"] = d["jcd"].astype(str).str.zfill(2)
        d["rno"] = d["rno"].astype(int)
    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    df = ent.merge(oof, on=["date", "jcd", "rno", "lane"], how="left")
    df = df.merge(mk, on=["date", "jcd", "rno", "lane"], how="left")
    df = df.merge(pay, on=["date", "jcd", "rno"], how="left")
    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


def report(df: pd.DataFrame, label: str) -> None:
    # モデル確率が6艇そろっとるレースだけ(欠けた艇があると argmax が偏る)
    ok6 = df.groupby(["date", "jcd", "rno"])["p"].transform(lambda s: s.notna().sum()) == 6
    n_drop = df.loc[~ok6].drop_duplicates(["date", "jcd", "rno"]).shape[0]
    df = df[ok6].copy()
    g = df.groupby(["date", "jcd", "rno"], sort=False)
    # レースごとに、3つの手が選ぶ艇番
    pick_lane1 = 1
    idx_model = g["p"].idxmax()
    idx_mark = df[df["mark"] == "◎"].set_index(["date", "jcd", "rno"])["lane"]
    races = df.drop_duplicates(["date", "jcd", "rno"]).set_index(["date", "jcd", "rno"])
    model_lane = df.loc[idx_model].set_index(["date", "jcd", "rno"])["lane"]
    win_lane = races["tansho_lane"]
    yen = races["tansho_yen"].astype(float)

    keep = win_lane.notna() & yen.notna() & model_lane.notna()
    keep &= races.index.isin(idx_mark.index)
    n = int(keep.sum())
    if n == 0:
        print(f"\n[{label}] 対象0レース")
        return
    wl = win_lane[keep].astype(float)
    y = yen[keep]
    picks = {
        "① 1号艇固定": pd.Series(pick_lane1, index=wl.index, dtype=float),
        "② モデル本命": model_lane[keep].astype(float),
        "③ 記者の◎": idx_mark.reindex(wl.index).astype(float),
    }
    # 🚨 「6艇フラット全買いの回収率」は**払戻率やない**。
    #    穴が過剰人気(w_i > p_i)やと p_i/w_i が1を割るんで、払戻率より下に出る。
    #    参考として出すだけで、到達率の分母には使わん。
    flat6 = float(y.sum() / (6 * 100 * n))
    # 払戻率は odds_timeseries の 1/Σ(1/オッズ) で別に実測した:
    #   全24場 6,568レース(締切−3分)で 中央 73.57%、場ごと 73.3〜73.7 とほぼ一定。
    #   表示オッズは切り捨てやから 1/オッズ が過大 → この推定は**低め**に出る。
    #   公称(控除25%)= 75% と整合するんで、到達率は 75% を分母にする。
    kaeshi = KAESHI
    print(f"\n=== {label}  {n:,}レース ===")
    print(f"  払戻率 {kaeshi:.0%}(到達率の分母) / 6艇フラット全買いの回収率 {flat6:.1%}"
          f" ← これは払戻率やない")
    print(f"  (モデル確率が6艇そろわんで落としたレース {n_drop:,})")
    rows = []
    for nm, pk in picks.items():
        hit = (pk.to_numpy() == wl.to_numpy())
        roi = float(np.where(hit, y.to_numpy(), 0.0).sum() / (100 * n))
        reach = (roi - kaeshi) / (1 - kaeshi)
        rows.append((nm, hit.mean(), roi, reach, hit))
        print(f"  {nm:12s} 本命的中 {hit.mean():7.3%}   単勝ROI {roi:7.2%}   到達率 {reach:+7.1%}")
    # 記者 vs モデルの食い違い
    m = picks["② モデル本命"].to_numpy()
    k = picks["③ 記者の◎"].to_numpy()
    same = (m == k)
    print(f"  一致率(モデル本命=記者◎) {same.mean():.1%}")
    for nm, pk in (("② モデル", m), ("③ 記者", k)):
        h = (pk[~same] == wl.to_numpy()[~same])
        print(f"     食い違った {int((~same).sum()):,}レースでの的中: {nm} {h.mean():.2%}")
    print(f"  記者の◎が1号艇やった率 {float((k == 1).mean()):.1%}"
          f" / モデル本命が1号艇 {float((m == 1).mean()):.1%}")
    # 日クラスタ bootstrap で ③−② の差
    dates = pd.Index(wl.index.get_level_values(0))
    _, dc = np.unique(dates, return_inverse=True)
    nd = dc.max() + 1
    d_hit = rows[2][4].astype(float) - rows[1][4].astype(float)
    s = np.bincount(dc, weights=d_hit, minlength=nd)
    rng = np.random.default_rng(20260924)
    boot = np.array([(s * (1 + rng.standard_normal(nd))).sum() / n for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  ③記者 − ②モデル の本命的中差 {d_hit.mean()*100:+.3f}pt "
          f"CI95 [{lo*100:+.3f}, {hi*100:+.3f}](日クラスタ)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--cutoff", default="2025")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    df = load(a.db)
    nr = df.drop_duplicates(["date", "jcd", "rno"]).shape[0]
    nm = df[df["mark"] == "◎"].shape[0]
    print(f"宮島 {nr:,}レース / 記者の◎が付いたレース {nm:,}")
    if a.dry:
        print("(dry: 数字は出さん)")
        return
    report(df, "全期間")
    report(df[df["date"].str[:4] < a.cutoff], f"訓練 (< {a.cutoff})")
    report(df[df["date"].str[:4] >= a.cutoff], f"holdout (>= {a.cutoff})")


if __name__ == "__main__":
    main()
