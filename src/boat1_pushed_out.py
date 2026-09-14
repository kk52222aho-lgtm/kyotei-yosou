"""1号艇が1コースを奪われるレースを事前に当てる。

`maegumi_detector` で分かったこと:
  1号艇が1コース  251,408レース (99.0%)  勝率 55.19%
  1号艇が2コース    2,137レース ( 0.8%)  勝率 **11.75%**
  1号艇が3コース      266レース ( 0.1%)  勝率 **11.28%**

落差が桁違いや。やが起きるんは 0.95% だけ = **極端に薄い的**。
十分位で切っても見えんから、上位1%・0.5%・0.1%の精度で測る。

作る特徴(全部そのレースより前のみ・リークなし):
  grab      2-6号艇それぞれの「過去に1コースを取った率」の最大/合計
            ← 1コースを奪いに来る奴が居るか。これが本体
  b1_out    1号艇の選手が過去に1枠から押し出された率
  b1_hold   1号艇の選手が過去に1枠から1コースを守った率
  press     内攻め率の合計(前と同じ)
  venue_out その場で1号艇が押し出される率(場×枠の前歴)

規律: walk-forward(学習=テスト年より前)。回収は確定払戻のみ。
     相手は必ず出す(同じレースで1号艇を素直に買う手)。

usage: python -u -m src.boat1_pushed_out
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import storage
from .maegumi_detector import prior_rate
from .venues import name as venue_name

RNG = np.random.default_rng(20260902)
NBOOT = 2000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.reg, e.wind_speed, e.wave_height,
               s.course, p.tansho_lane, p.tansho_yen,
               p.trifecta_combo, p.trifecta_yen, p.exacta_combo, p.exacta_yen
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
    df["grab1"] = (df["course"] == 1).astype(float)          # 1コースを取った
    df["in_"] = (df["course"] < df["lane"]).astype(float)
    df["out1"] = ((df["lane"] == 1) & (df["course"] > 1)).astype(float)
    df["hold1"] = ((df["lane"] == 1) & (df["course"] == 1)).astype(float)
    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


def main():
    df = load()
    print(f"盤: {len(df):,}行 / {df['rid'].nunique():,}レース")

    df["p_grab"] = prior_rate(df, ["reg", "lane"], "grab1")   # その枠から1コース取る率
    df["p_in"] = prior_rate(df, ["reg", "lane"], "in_")
    df["p_out1"] = prior_rate(df, ["reg"], "out1")
    df["p_hold1"] = prior_rate(df, ["reg"], "hold1")
    df["v_out"] = prior_rate(df, ["jcd"], "out1")

    outer = df[df["lane"] >= 2]
    G = outer.groupby("rid")["p_grab"].agg(grab_max="max", grab_sum="sum")
    P = outer.groupby("rid")["p_in"].agg(press_max="max", press="sum")
    b1 = df[df["lane"] == 1].set_index("rid")

    R = pd.DataFrame(index=b1.index)
    R = R.join(G).join(P)
    R["b1_out"] = b1["p_out1"]
    R["b1_hold"] = b1["p_hold1"]
    R["v_out"] = b1["v_out"]
    R["wind"] = b1["wind_speed"]
    R["wave"] = b1["wave_height"]
    R["date"] = b1["date"]
    R["jcd"] = b1["jcd"]
    R["yr"] = b1["yr"]
    R["c1"] = b1["course"]
    R["tansho_lane"] = b1["tansho_lane"]
    R["tansho_yen"] = b1["tansho_yen"]
    R["trifecta_combo"] = b1["trifecta_combo"]
    R["trifecta_yen"] = b1["trifecta_yen"]
    R = R.reset_index().rename(columns={"index": "rid"})
    R["y"] = (R["c1"] > 1).astype(int)
    R["win1"] = (R["tansho_lane"] == 1).astype(int)
    FEA = ["grab_max", "grab_sum", "press_max", "press", "b1_out", "b1_hold",
           "v_out", "wind", "wave"]
    R = R.dropna(subset=FEA).reset_index(drop=True)
    print(f"特徴がそろうレース: {len(R):,}   1号艇が押し出される率 {R['y'].mean():.2%}")

    X = R[FEA].to_numpy(float)
    y = R["y"].to_numpy(int)
    pr = np.full(len(R), np.nan)
    for yv in sorted(R["yr"].unique()):
        te = (R["yr"] == yv).to_numpy()
        tr = (R["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 5000 or te.sum() < 2000:
            continue
        print(f"[fit] {yv} train={tr.sum():,}", flush=True)
        m = CalibratedClassifierCV(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0,
            random_state=42), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        pr[te] = m.predict_proba(X[te])[:, 1]
    R["p"] = pr
    d = R.dropna(subset=["p"]).sort_values("p", ascending=False).reset_index(drop=True)
    base = d["y"].mean()
    print(f"\n=== ① 1号艇が押し出されるのを当てられるか ===")
    print(f"  AUC {roc_auc_score(d['y'], d['p']):.4f}   ベース(全レース) {base:.2%}")
    print(f"\n  {'上位':<10}{'レース':>8}{'押出し率':>10}{'ベース比':>9}"
          f"{'1号艇勝率':>11}{'1号艇単勝回収':>13}")
    for frac in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        k = max(int(len(d) * frac), 50)
        g = d.head(k)
        h = g["win1"] == 1
        roi = g.loc[h, "tansho_yen"].sum() / (len(g) * 100)
        print(f"  上位{frac:<7.1%}{k:>8,}{g['y'].mean():>10.2%}"
              f"{g['y'].mean() / base:>8.1f}x{g['win1'].mean():>11.2%}{roi:>13.1%}")
    hall = d["win1"] == 1
    print(f"  {'全体':<10}{len(d):>8,}{base:>10.2%}{1.0:>8.1f}x"
          f"{d['win1'].mean():>11.2%}"
          f"{d.loc[hall, 'tansho_yen'].sum() / (len(d) * 100):>13.1%}")

    print(f"\n=== ② 実際に押し出されたレースだけ(答えを見た上限) ===")
    a = d[d["y"] == 1]
    ha = a["win1"] == 1
    print(f"  {len(a):,}レース  1号艇勝率 {a['win1'].mean():.2%}  "
          f"1号艇単勝回収 {a.loc[ha, 'tansho_yen'].sum() / (len(a) * 100):.1%}")
    print(f"  ※ ここが天井。事前に完全に当てられてもこれ以上は出ん")

    print(f"\n=== ③ 上位1%で1号艇を切って買う(確定払戻) ===")
    k = int(len(d) * 0.01)
    g = d.head(k).copy()
    print(f"  対象 {len(g):,}レース")
    import itertools
    sets = {
        "1号艇 単勝(相手)": None,
        "2号艇 単勝": 2, "3号艇 単勝": 3, "4号艇 単勝": 4,
    }
    for lab, ln in sets.items():
        if ln is None:
            h = g["win1"] == 1
            print(f"  {lab:<20}的中 {h.mean():>6.2%}  "
                  f"回収 {g.loc[h, 'tansho_yen'].sum() / (len(g) * 100):>6.1%}")
        else:
            h = g["tansho_lane"] == ln
            print(f"  {lab:<20}的中 {h.mean():>6.2%}  "
                  f"回収 {g.loc[h, 'tansho_yen'].sum() / (len(g) * 100):>6.1%}")
    heads = [f"{a_}-{b_}-{c_}" for a_ in (2, 3, 4)
             for b_, c_ in itertools.permutations([x for x in range(1, 7)
                                                   if x != a_], 2)]
    hit = g["trifecta_combo"].isin(heads)
    roi = g.loc[hit, "trifecta_yen"].sum() / (len(g) * len(heads) * 100)
    ud, iv = np.unique(g["date"], return_inverse=True)
    w = np.bincount(iv, weights=np.where(hit, g["trifecta_yen"], 0.0), minlength=len(ud))
    s = np.bincount(iv, minlength=len(ud)) * len(heads) * 100
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    lo, hi = np.percentile(w[idx].sum(1) / s[idx].sum(1), [2.5, 97.5])
    print(f"  3連単 頭2,3,4 全通り({len(heads)}点)  的中 {hit.mean():>6.2%}  "
          f"回収 {roi:>6.1%} CI[{lo:.1%},{hi:.1%}]")
    R.to_csv("data/boat1_pushed.csv", index=False, encoding="utf-8-sig")
    print("\n保存: data/boat1_pushed.csv")


if __name__ == "__main__":
    main()
