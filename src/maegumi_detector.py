"""「1号艇が飛ぶレース」の検出器。選手の前づけ癖を過去151万行から作る。

背景: モデルは 91.2% のレースで「1号艇」と言う一辺倒やった。原因は採点を
「誰が1着か」にしとったから(1号艇が55%勝つ競技では argmax がほぼ常に1号艇になる)。

今日 K ファイルから掘った `results_st`(進入コース 151万行)で分かったこと:
  枠なり      79.9%  → 1号艇勝率 56.77%
  前づけあり   20.1%  → 1号艇勝率 46.76%
  1号艇が2コース 0.8% → 1号艇勝率 11.75%

進入コースはレース後確定やが、**選手ごとの前づけ癖は出走表の時点で分かる。**
過去のそのレースより前の進入履歴だけで作る(リークなし)。

作る特徴(全部そのレースより前のみ):
  self_in    その選手がその枠から内に入った率
  self_out   その選手がその枠から外に出された率
  press      2-6号艇の内攻め率の合計 = レースにかかる前づけ圧
  press_max  外(4-6号艇)の中で一番内に来る奴の率
  venue_mg   その場の直近の前づけ率
  boat1_out  1号艇の選手が押し出された率

問い:
  ① 進入が崩れるレースを事前に当てられるか
  ② 「1号艇が飛ぶ」を当てられるか。**1号艇ベタ(54.76%)を超えるか**
  ③ 当てた上で、そこで単勝が買えるか(確定払戻)

usage: python -u -m src.maegumi_detector
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from . import storage
from .venues import name as venue_name

RNG = np.random.default_rng(20260902)
NBOOT = 2000
MIN_OBS = 5


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.reg, e.finish, e.racer_class,
               e.nat_win, e.nat_2rate, e.loc_win, e.motor_2rate, e.tenji_time,
               e.wind_speed, e.wave_height,
               s.course, s.st, s.st_f,
               p.tansho_lane, p.tansho_yen
        FROM entries e
        JOIN results_st s ON e.date=s.date AND e.jcd=s.jcd AND e.rno=s.rno AND e.lane=s.lane
        JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    sz = df.groupby(["date", "jcd", "rno"])["lane"].transform("size")
    df = df[sz == 6].copy()
    df["rid"] = df["date"] + df["jcd"] + df["rno"].astype(str).str.zfill(2)
    df["yr"] = df["date"].str[:4]
    df["in_"] = (df["course"] < df["lane"]).astype(float)     # 内に入った
    df["out_"] = (df["course"] > df["lane"]).astype(float)    # 外へ出された
    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


def prior_rate(df, keys, col):
    """そのレースより前だけの平均(リークなし)。観測 MIN_OBS 未満は NaN。"""
    ka = [pd.factorize(df[c])[0] for c in keys]
    order = np.lexsort([df["lane"].to_numpy(), df["rno"].to_numpy(),
                        df["date"].to_numpy()] + ka[::-1])
    v = df[col].to_numpy(float)[order]
    ok = ~np.isnan(v)
    CS = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    CM = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    kk = np.column_stack([a[order] for a in ka])
    newg = np.ones(len(v), bool)
    newg[1:] = (kk[1:] != kk[:-1]).any(axis=1)
    gs = np.flatnonzero(newg)[np.cumsum(newg) - 1]
    i = np.arange(len(v))
    s, c = CS[i] - CS[gs], CM[i] - CM[gs]
    out = np.where(c >= MIN_OBS, s / np.where(c == 0, 1, c), np.nan)
    inv = np.empty(len(v), np.int64)
    inv[order] = i
    return out[inv]


def main():
    df = load()
    print(f"盤: {len(df):,}行 / {df['rid'].nunique():,}レース "
          f"({df['date'].min()}-{df['date'].max()})")

    df["self_in"] = prior_rate(df, ["reg", "lane"], "in_")
    df["self_out"] = prior_rate(df, ["reg", "lane"], "out_")
    df["self_in_any"] = prior_rate(df, ["reg"], "in_")
    df["venue_mg"] = prior_rate(df, ["jcd", "lane"], "in_")

    g = df.groupby("rid", sort=False)
    outer = df[df["lane"] >= 4]
    df = df.merge(outer.groupby("rid")["self_in"].max().rename("press_max"),
                  on="rid", how="left")
    df = df.merge(df[df["lane"] >= 2].groupby("rid")["self_in"].sum()
                  .rename("press"), on="rid", how="left")
    b1 = df[df["lane"] == 1][["rid", "self_out", "self_in_any"]].rename(
        columns={"self_out": "boat1_out", "self_in_any": "boat1_agg"})
    df = df.merge(b1, on="rid", how="left")

    # レース単位の盤
    R = df[df["lane"] == 1][["rid", "date", "jcd", "yr", "rno", "tansho_lane",
                             "tansho_yen", "press", "press_max", "boat1_out",
                             "venue_mg", "wind_speed", "wave_height"]].copy()
    R["mg"] = df.groupby("rid")["in_"].max().reindex(R["rid"]).to_numpy()
    R["c1"] = df[df.lane == 1].set_index("rid")["course"].reindex(R["rid"]).to_numpy()
    R["win1"] = (R["tansho_lane"] == 1).astype(int)
    R["場"] = R["jcd"].map(venue_name)
    R = R.dropna(subset=["press", "press_max", "boat1_out", "venue_mg"])
    print(f"特徴がそろうレース: {len(R):,}  (前づけ癖の履歴が要るので序盤が落ちる)")

    FEA = ["press", "press_max", "boat1_out", "venue_mg", "wind_speed", "wave_height"]
    X = R[FEA].to_numpy(float)

    def walk(y):
        pr = np.full(len(R), np.nan)
        for yv in sorted(R["yr"].unique()):
            te = (R["yr"] == yv).to_numpy()
            tr = (R["yr"].astype(int) < int(yv)).to_numpy()
            if tr.sum() < 5000 or te.sum() < 2000:
                continue
            m = CalibratedClassifierCV(HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.08, max_iter=300,
                l2_regularization=1.0, random_state=42), method="isotonic", cv=3)
            m.fit(X[tr], y[tr])
            pr[te] = m.predict_proba(X[te])[:, 1]
        return pr

    print("\n① 進入が崩れるレースを事前に当てられるか", flush=True)
    R["p_mg"] = walk(R["mg"].to_numpy(int))
    d = R.dropna(subset=["p_mg"])
    from sklearn.metrics import roc_auc_score
    print(f"  AUC {roc_auc_score(d['mg'], d['p_mg']):.4f}   実際の前づけ率 {d['mg'].mean():.1%}")
    d = d.copy()
    d["dec"] = pd.qcut(d["p_mg"], 10, labels=False, duplicates="drop")
    t = d.groupby("dec").agg(n=("mg", "size"), 予測=("p_mg", "mean"),
                             実際=("mg", "mean"), 一号艇勝率=("win1", "mean"))
    print(f"  {'十分位':<6}{'レース':>8}{'予測':>8}{'実際':>8}{'1号艇勝率':>10}")
    for k, r in t.iterrows():
        print(f"  {int(k) + 1:<6}{int(r['n']):>8,}{r['予測']:>8.1%}"
              f"{r['実際']:>8.1%}{r['一号艇勝率']:>10.2%}")

    print("\n② 1号艇が飛ぶかを当てられるか(1号艇ベタ = 全体の勝率が相手)", flush=True)
    R["p_w1"] = walk(R["win1"].to_numpy(int))
    d = R.dropna(subset=["p_w1"]).copy()
    base = d["win1"].mean()
    print(f"  AUC {roc_auc_score(d['win1'], d['p_w1']):.4f}   1号艇ベタ {base:.2%}")
    d["dec"] = pd.qcut(d["p_w1"], 10, labels=False, duplicates="drop")
    t = d.groupby("dec").agg(n=("win1", "size"), 予測=("p_w1", "mean"),
                             実際=("win1", "mean"), 前づけ率=("mg", "mean"),
                             圧=("press", "mean"))
    print(f"  {'十分位':<6}{'レース':>8}{'予測':>8}{'実際':>9}{'前づけ率':>9}{'圧':>7}")
    for k, r in t.iterrows():
        print(f"  {int(k) + 1:<6}{int(r['n']):>8,}{r['予測']:>8.1%}"
              f"{r['実際']:>9.2%}{r['前づけ率']:>9.1%}{r['圧']:>7.2f}")
    lo = d[d["dec"] == 0]
    ud, iv = np.unique(lo["date"], return_inverse=True)
    s = np.bincount(iv, weights=lo["win1"].to_numpy(float), minlength=len(ud))
    c = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    b = s[idx].sum(1) / c[idx].sum(1)
    q = np.percentile(b, [2.5, 97.5])
    print(f"\n  最下位十分位(1号艇が一番危ないと言うたレース) {len(lo):,}本")
    print(f"    1号艇勝率 {lo['win1'].mean():.2%} CI[{q[0]:.2%},{q[1]:.2%}]"
          f"  vs 全体 {base:.2%}  差 {(lo['win1'].mean() - base) * 100:+.2f}pt")

    print("\n③ そこで1号艇を切ったら単勝は買えるか", flush=True)
    tan = df.merge(lo[["rid"]], on="rid")
    tan = tan[tan["lane"] != 1]
    hit = tan["lane"] == tan["tansho_lane"]
    print(f"  1号艇以外5艇を全部買う: {len(tan):,}点  的中 {hit.mean():.2%}  "
          f"回収 {tan.loc[hit, 'tansho_yen'].sum() / (len(tan) * 100):.1%}")
    for L in (2, 3, 4, 5, 6):
        t2 = tan[tan["lane"] == L]
        h2 = t2["lane"] == t2["tansho_lane"]
        print(f"    {L}号艇単勝: {len(t2):>6,}点  的中 {h2.mean():>6.2%}  "
              f"回収 {t2.loc[h2, 'tansho_yen'].sum() / (len(t2) * 100):>6.1%}")
    R.to_csv("data/maegumi_races.csv", index=False, encoding="utf-8-sig")
    print("\n保存: data/maegumi_races.csv")


if __name__ == "__main__":
    main()
