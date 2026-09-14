"""走査24本の正体を割る対照 — 分母を「既存14を使い切った版」にする。

=== なぜこの対照が要るか(2026-09-02) ===
偽薬で分かったんは「**既存モデルが自分の特徴のレース内順位を使い切っとらん**」やった
(モデルに入っとる nat_win が、そのモデルの残差に増分 t=+17.0)。

走査の測定側はそれを直交化で潰した。やが **モデル側の手抜きは直っとらんまま**や。
そやから `既存14 → 既存14+走査24 = +0.28pt` は、まだ2つを区別できてへん:

  (a) 走査24本が本物の新情報
  (b) 走査24本の正体が「既存14の順位情報の穴埋め」= 5行の特徴量追加でタダで取れる

正しい分母は既存14やのうて、**既存14 + その14本のレース内順位**や。

  ほぼ残る    → (a) 走査は本物の新情報
  ほとんど消える → (b) 自分のモデルの手抜きを埋めとっただけ

ノイズ24列の対照よりこっちの方が答えが変わる。

=== 並べる4つ(全部 同じレース集合・同じ walk-forward) ===
  A 1号艇固定(予測が要らん手)
  B 既存14特徴                      ← 今までの分母
  C **既存14 + 順位14(使い切った版)** ← 正しい分母
  D 既存14 + 順位14 + 走査24

順位を取る14本は、走査の直交化で使うたんと**同じ列**にする
(venue_code はカテゴリ、wind/wave はレース内で定数、展開4本は歴史データで定数なので除く)。
順位は `pct=True`(レース内の相対位置)。判定は **D − C**。

採点年は holdout(走査が一度も見てへん 2025/2026)だけ。

usage: python -u -m src.test_saturated_base
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .scan_all import KEYS, lagged, load as scan_load
from .train import _make_estimator

RNG = np.random.default_rng(20260902)
NBOOT = 2000
N_MAX = 24
HOLDOUT = ["2025", "2026"]

# 走査の直交化で使うたんと同じ14本
BASE_RANK = ["lane", "class_num", "age", "weight",
             "nat_win", "nat_2rate", "loc_win", "loc_2rate",
             "motor_2rate", "boat_2rate", "tenji_time",
             "tt_rank", "tt_gap", "inner_pow_edge"]


def cluster_ci(a, b, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    sa = np.bincount(iv, weights=a, minlength=len(ud))
    sb = np.bincount(iv, weights=b, minlength=len(ud))
    cn = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    c = cn[idx].sum(1)
    return np.percentile(sa[idx].sum(1) / c - sb[idx].sum(1) / c, [2.5, 97.5])


def main():
    res = pd.read_csv("data/scan_holdout.csv")     # 2023-24 だけで走査したもん
    sur = res[res["survive_ort"]].copy()
    sur[["key", "qty", "lag"]] = sur["feature"].str.split("|", expand=True)
    sur = sur.reindex(sur["t_ort"].abs().sort_values(ascending=False).index)
    kept = sur.drop_duplicates(subset=["key", "qty"], keep="first").head(N_MAX)
    print(f"走査24本(holdout選抜・2025/26 は一度も見てへん): {len(kept)}本")

    df = scan_load()
    extra = []
    for _, r in kept.iterrows():
        for lname, vals in lagged(df, KEYS[r["key"]], r["qty"]):
            if lname == r["lag"]:
                nm = f"S_{r['key']}_{r['qty']}_{r['lag']}"
                df[nm] = vals
                extra.append(nm)
                break
    assert len(extra) == len(kept)

    conn = storage.connect()
    raw = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    pay = pd.read_sql_query("SELECT date, jcd, rno, tansho_lane FROM payouts "
                            "WHERE tansho_lane IS NOT NULL", conn)
    conn.close()
    raw["jcd"] = raw["jcd"].astype(str).str.zfill(2)
    pay["jcd"] = pay["jcd"].astype(str).str.zfill(2)

    F = build_frame(raw, impute=False)
    keys = ["date", "jcd", "rno", "lane"]
    F[keys] = raw[keys]
    F["win"] = raw["win"].to_numpy()
    F = F.merge(df[keys + extra], on=keys, how="left")
    F = F.merge(pay, on=["date", "jcd", "rno"], how="inner")
    F["yr"] = F["date"].str[:4]

    # ---- 使い切った版: 既存14のレース内順位 ----
    g = F.groupby(["date", "jcd", "rno"], sort=False)
    ranks = []
    for c in BASE_RANK:
        nm = f"R_{c}"
        F[nm] = g[c].rank(pct=True)      # レース内の相対位置(レース頭数に依らん)
        ranks.append(nm)
    print(f"順位列 {len(ranks)}本を追加。非欠測 "
          f"{F[ranks].notna().all(axis=1).mean():.1%} / 走査列 "
          f"{F[extra].notna().all(axis=1).mean():.1%}")

    y = F["win"].to_numpy(dtype=int)
    sets = {
        "B 既存14": FEATURES,
        "C 既存14+順位14": FEATURES + ranks,
        "D 既存14+順位14+走査24": FEATURES + ranks + extra,
    }
    preds = {}
    for name, cols in sets.items():
        X = F[cols].to_numpy(dtype=float)
        out = []
        for yv in sorted(F["yr"].unique()):
            te = (F["yr"] == yv).to_numpy()
            tr = (F["yr"].astype(int) < int(yv)).to_numpy()
            if tr.sum() < 3000 or te.sum() < 2000:
                continue
            print(f"[fit] {name:20s} {yv} train={tr.sum():,} 特徴{len(cols)}", flush=True)
            est = _make_estimator()
            est.categorical_features = [cols.index("venue_code")]
            mdl = CalibratedClassifierCV(est, method="isotonic", cv=3)
            mdl.fit(X[tr], y[tr])
            o = F.loc[te, keys + ["tansho_lane", "yr"]].copy()
            o["p"] = mdl.predict_proba(X[te])[:, 1]
            out.append(o)
        preds[name] = pd.concat(out).set_index(keys)

    j = preds["B 既存14"][["p", "tansho_lane", "yr"]].rename(columns={"p": "pB"})
    j = j.join(preds["C 既存14+順位14"][["p"]].rename(columns={"p": "pC"}), how="inner")
    j = j.join(preds["D 既存14+順位14+走査24"][["p"]].rename(columns={"p": "pD"}),
               how="inner").reset_index()
    rows = []
    for _, gg in j.groupby(["date", "jcd", "rno"], sort=False):
        if len(gg) != 6:
            continue
        wl = int(gg["tansho_lane"].iloc[0])
        rows.append({
            "date": gg["date"].iloc[0], "yr": gg["yr"].iloc[0],
            "lane1": int(wl == 1),
            "B": int(int(gg.loc[gg["pB"].idxmax(), "lane"]) == wl),
            "C": int(int(gg.loc[gg["pC"].idxmax(), "lane"]) == wl),
            "D": int(int(gg.loc[gg["pD"].idxmax(), "lane"]) == wl),
        })
    R = pd.DataFrame(rows)
    R.to_csv("data/saturated_base_races.csv", index=False)

    def rep(sub, label):
        print(f"\n=== {label}  n={len(sub):,}レース / {sub['date'].nunique():,}日 ===")
        d = sub["date"].to_numpy()
        for nm, c in [("A 1号艇固定(予測不要)", "lane1"), ("B 既存14", "B"),
                      ("C 既存14+順位14", "C"), ("D +走査24", "D")]:
            print(f"  {nm:22s} 本命的中 {sub[c].mean():.3%}")
        print()
        for nm, a, b in [("B − A(既存 vs 予測不要)", "B", "lane1"),
                         ("C − A(使い切った版 vs 予測不要)", "C", "lane1"),
                         ("D − A(全部乗せ vs 予測不要)", "D", "lane1"),
                         ("**C − B(順位を足しただけ)**", "C", "B"),
                         ("**D − C(走査の真の上乗せ)**", "D", "C"),
                         ("D − B(旧い測り方)", "D", "B")]:
            lo, hi = cluster_ci(sub[a].to_numpy(float), sub[b].to_numpy(float), d)
            gap = sub[a].mean() - sub[b].mean()
            v = "効いとる" if lo > 0 else ("悪化" if hi < 0 else "ゼロを跨いどる")
            print(f"  {nm:34s} {gap * 100:+.2f}pt  "
                  f"CI[{lo * 100:+.2f},{hi * 100:+.2f}]  ← {v}")

    rep(R[R["yr"].isin(HOLDOUT)], "holdout 2025/2026(走査が一度も見てへん年)")
    rep(R, "全年(参考)")


if __name__ == "__main__":
    main()
