"""看板 "AUC 0.83 / 本命54.9% > ベース53.6%" を、予測が要らん手と同じ土俵で測り直す。

=== なぜ測り直すか ===
看板の数字は 12,516レース・期間外test **1,806レース** で出したもん(docs/FINDINGS.md A1)。
今 DB には **253,035レース**ある。20倍や。しかも 53.6% は当時のtest窓の1号艇ベタ買いで、
その後の test_ablation.py では **定数として貼り付けられとる**(別サンプルの数字を今の
walk-forward精度と並べとる)。同じ窓・同じレースで並べ直さんと差は測れん。

=== 数字を見る前に決めた設計(2026-09-02) ===
(1) 並べる相手は3つ([[principle-dumb-baseline]]):
      A 予測が要らん手       = 毎レース1号艇
      B その分野の標準推定量 = 公式の全国勝率が最上位の艇(誰でもタダで見れる)
      C 素朴な測り方         = 艇番だけを特徴にしたモデル(=事実上A、整合性の点呼用)
(2) **対で測る。**同じレース集合の上で model_hit と base_hit を並べ、差の CI を出す。
    それぞれ別々に「54.9%」「53.6%」と書くのは比較やない。
(3) CI は **日付クラスタのブートストラップ**。同じ日・同じ場のレースは独立やない。
    加えて McNemar(不一致ペアのみ)を出す。対で見ると分母が激減するのが肝や。
(4) walk-forward(学習=テスト年より前のみ)。特徴選択に 2022/2026 を使うとるので
    クリーン年(2023-2025)と全年の両方を出す。
(5) 6艇そろいのレースのみ(飛び艇の survivorship 除去)。

usage: python -u -m src.test_dumb_base
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import storage
from .features import FEATURES, CAT_INDEX, build_frame
from .validate import SELECTION_YEARS

RNG = np.random.default_rng(20260902)
NBOOT = 2000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.reg, e.racer_class, e.age, e.weight,
               e.avg_st, e.nat_win, e.nat_2rate, e.loc_win, e.loc_2rate,
               e.motor_2rate, e.boat_2rate, e.tenji_time, e.tenji_st,
               e.wind_speed, e.wave_height, e.win, e.finish,
               p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def est(cat_idx):
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0,
        categorical_features=([cat_idx] if cat_idx is not None else []), random_state=42)


def run_year(full, y, df, tr, te, cols, cat_idx):
    m = CalibratedClassifierCV(est(cat_idx), method="isotonic", cv=3)
    m.fit(full[cols].to_numpy(dtype=float)[tr], y[tr])
    return m.predict_proba(full[cols].to_numpy(dtype=float)[te])[:, 1]


def main():
    df = load()
    full = build_frame(df, impute=False)[FEATURES].reset_index(drop=True)
    y = df["win"].to_numpy(dtype=int)
    df = df.reset_index(drop=True)

    years = [v for v in sorted(df["yr"].unique()) if v > min(df["yr"])]
    rows = []
    aucs = []
    for yv in years:
        te = (df["yr"] == yv).to_numpy()
        tr = (df["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            print(f"[skip] {yv}: train={tr.sum()} test={te.sum()}")
            continue
        print(f"[fit] {yv}  train={tr.sum():,} test={te.sum():,}", flush=True)
        p_full = run_year(full, y, df, tr, te, FEATURES, FEATURES.index("venue_code"))
        p_lane = run_year(full, y, df, tr, te, ["lane"], None)   # C: 艇番だけ
        out = df[te].copy()
        out["p"] = p_full
        out["p_lane"] = p_lane
        aucs.append((yv, roc_auc_score(y[te], p_full), roc_auc_score(y[te], p_lane)))
        for (d, j, r), g in out.groupby(["date", "jcd", "rno"], sort=False):
            if len(g) != 6:
                continue
            wl = int(g["tansho_lane"].iloc[0])
            gm = g.sort_values("p", ascending=False)
            gl = g.sort_values("p_lane", ascending=False)
            gn = g.sort_values("nat_win", ascending=False)   # B: 公式全国勝率
            rows.append({
                "date": d, "yr": yv,
                "model": int(int(gm.iloc[0]["lane"]) == wl),
                "lane1": int(wl == 1),
                "natwin": int(int(gn.iloc[0]["lane"]) == wl) if pd.notna(gn.iloc[0]["nat_win"]) else np.nan,
                "laneonly": int(int(gl.iloc[0]["lane"]) == wl),
                "top_is_1": int(int(gm.iloc[0]["lane"]) == 1),
            })
    R = pd.DataFrame(rows)
    R.to_csv("data/dumb_base_races.csv", index=False)

    print("\n=== AUC (walk-forward) ===")
    for yv, a1, a2 in aucs:
        print(f"  {yv}  全特徴 {a1:.4f}   艇番だけ {a2:.4f}")

    def report(sub, label):
        n = len(sub)
        print(f"\n=== {label}  n={n:,}レース  日数={sub['date'].nunique():,} ===")
        print(f"  モデル本命が1号艇やった率: {sub['top_is_1'].mean():.1%}")
        dates = sub["date"].to_numpy()
        udates, inv = np.unique(dates, return_inverse=True)
        nd = len(udates)
        for col, name in [("lane1", "A 1号艇固定(予測不要)"),
                          ("natwin", "B 全国勝率トップ(公式・無料)"),
                          ("laneonly", "C 艇番だけモデル")]:
            s = sub[["model", col]].dropna()
            if not len(s):
                continue
            m_, b_ = s["model"].to_numpy(), s[col].to_numpy()
            gap = m_.mean() - b_.mean()
            # 日付クラスタ bootstrap
            d2 = sub.loc[s.index, "date"].to_numpy()
            ud, iv = np.unique(d2, return_inverse=True)
            sm = np.bincount(iv, weights=m_, minlength=len(ud))
            sb = np.bincount(iv, weights=b_, minlength=len(ud))
            cnt = np.bincount(iv, minlength=len(ud))
            idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
            bm, bb, bc = sm[idx].sum(1), sb[idx].sum(1), cnt[idx].sum(1)
            boot = bm / bc - bb / bc
            lo, hi = np.percentile(boot, [2.5, 97.5])
            # McNemar
            bwin = int(((m_ == 1) & (b_ == 0)).sum())
            cwin = int(((m_ == 0) & (b_ == 1)).sum())
            disc = bwin + cwin
            z = (bwin - cwin) / np.sqrt(disc) if disc else float("nan")
            print(f"  {name}")
            print(f"    モデル {m_.mean():.3%}  vs  相手 {b_.mean():.3%}   "
                  f"差 {gap*100:+.2f}pt  CI[{lo*100:+.2f},{hi*100:+.2f}]pt")
            print(f"    不一致ペア {disc:,}件 (モデルだけ的中 {bwin:,} / 相手だけ的中 {cwin:,})  "
                  f"McNemar z={z:+.2f}")

    report(R, "全年")
    report(R[~R["yr"].isin(SELECTION_YEARS)], "クリーン年のみ(特徴選択に未使用)")


if __name__ == "__main__":
    main()
