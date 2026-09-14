"""走査の生き残りを **モデルに載せて、1号艇固定と同じ土俵で** 並べ直す確認工程。

走査(scan_all.py)が出すのは「既存情報を線形順位で使い切った残差とも相関しとる」いう
**候補**や。それは「本命的中率が上がる」とも「1号艇固定との差が広がる」とも別の話や。
[[principle-dumb-baseline]] の通り、出口には必ず予測が要らん手を置く。

=== 数字を見る前に決めた手順(2026-09-02) ===
(1) 直交化残差の生き残り(|t_ort| > FWER閾値)を取る。
(2) **同じ(キー,量)の中はラグ1本に絞る**(|t_ort|最大)。窓とラグは互いにほぼ同じ列や。
    共線な列を50本突っ込んでも木が薄く割れるだけ。
(3) **上限 N=24 本**(|t_ort|降順)。落とした分は本数と中身を必ず出す。黙って切らん。
    24 は「既存14特徴と同じオーダー」いう理由で先に決めた。結果を見て動かさん。
(4) walk-forward(学習=テスト年より前)。全6艇そろい・確定単勝で採点。欠測は補完せん
    (HistGBM がネイティブに扱う)。埋めたら「その選手に今節の履歴が無い」が
    「平均的な成績」に化ける。
(5) 出す数字は3つを **同じレース集合の上で**:
      1号艇固定(予測不要) / 既存14特徴 / 既存+走査の生き残り
    **判定は「1号艇固定との差が広がったか」。**既存モデルに勝っただけでは通さん。
(6) CI は日付クラスタのブートストラップ。クリーン年(特徴選択に未使用)も別掲。

usage: python -u -m src.test_scan_confirm
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .scan_all import KEYS, lagged, load as scan_load
from .train import _make_estimator
from .validate import SELECTION_YEARS

RNG = np.random.default_rng(20260902)
NBOOT = 2000
N_MAX = 24


def pick(res: pd.DataFrame):
    sur = res[res["survive_ort"]].copy()
    sur[["key", "qty", "lag"]] = sur["feature"].str.split("|", expand=True)
    sur = sur.reindex(sur["t_ort"].abs().sort_values(ascending=False).index)
    one = sur.drop_duplicates(subset=["key", "qty"], keep="first")
    kept = one.head(N_MAX)
    print(f"直交化で生き残り {len(sur)}本 → (キー,量)ごと1本で {len(one)}本 "
          f"→ 上限{N_MAX}で {len(kept)}本")
    drop = one.iloc[N_MAX:]
    if len(drop):
        print(f"  ⚠ 上限で落としたもん {len(drop)}本(黙って切らんために全部出す):")
        for _, r in drop.iterrows():
            print(f"     {r['feature']:32s} t_ort={r['t_ort']:+7.2f}")
    return kept


def cluster_ci(a, b, dates):
    ud, iv = np.unique(dates, return_inverse=True)
    sa = np.bincount(iv, weights=a, minlength=len(ud))
    sb = np.bincount(iv, weights=b, minlength=len(ud))
    cn = np.bincount(iv, minlength=len(ud))
    idx = RNG.integers(0, len(ud), size=(NBOOT, len(ud)))
    c = cn[idx].sum(1)
    boot = sa[idx].sum(1) / c - sb[idx].sum(1) / c
    return np.percentile(boot, [2.5, 97.5])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/scan_all.csv"
    only = sys.argv[2:] or None        # 採点する年を絞る(holdout用)
    print(f"選抜元: {src}" + (f"  / 採点年を {only} に限定" if only else ""))
    res = pd.read_csv(src)
    if not res["survive_ort"].any():
        print("直交化残差の生き残りゼロ。載せるもんが無い。")
        return
    kept = pick(res)
    print("\n載せる特徴:")
    for _, r in kept.iterrows():
        print(f"  {r['feature']:34s} t_ort={r['t_ort']:+7.2f}  "
              f"使えたレース {int(r['races']):,}")

    df = scan_load()
    extra = []
    for _, r in kept.iterrows():
        for lname, vals in lagged(df, KEYS[r["key"]], r["qty"]):
            if lname == r["lag"]:
                name = f"S_{r['key']}_{r['qty']}_{r['lag']}"
                df[name] = vals
                extra.append(name)
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
    n0 = len(F)
    F = F.merge(df[keys + extra], on=keys, how="left")
    F = F.merge(pay, on=["date", "jcd", "rno"], how="inner")
    assert len(F) <= n0, "merge で行が増えた"
    F["yr"] = F["date"].str[:4]
    y = F["win"].to_numpy(dtype=int)
    print(f"\n採点盤: {len(F):,}行  走査特徴の非欠測率 "
          f"{F[extra].notna().all(axis=1).mean():.1%}")

    preds = {}
    for name, cols in {"既存14特徴": FEATURES, "既存+走査": FEATURES + extra}.items():
        X = F[cols].to_numpy(dtype=float)
        out = []
        for yv in sorted(F["yr"].unique()):
            te = (F["yr"] == yv).to_numpy()
            tr = (F["yr"].astype(int) < int(yv)).to_numpy()
            if tr.sum() < 3000 or te.sum() < 2000:
                continue
            print(f"[fit] {name:10s} {yv} train={tr.sum():,}", flush=True)
            est = _make_estimator()
            est.categorical_features = [cols.index("venue_code")]
            mdl = CalibratedClassifierCV(est, method="isotonic", cv=3)
            mdl.fit(X[tr], y[tr])
            o = F.loc[te, keys + ["tansho_lane", "yr"]].copy()
            o["p"] = mdl.predict_proba(X[te])[:, 1]
            out.append(o)
        preds[name] = pd.concat(out).set_index(keys)

    j = (preds["既存14特徴"][["p", "tansho_lane", "yr"]]
         .join(preds["既存+走査"][["p"]], rsuffix="_s", how="inner").reset_index())
    rows = []
    for _, g in j.groupby(["date", "jcd", "rno"], sort=False):
        if len(g) != 6:
            continue
        wl = int(g["tansho_lane"].iloc[0])
        rows.append({
            "date": g["date"].iloc[0], "yr": g["yr"].iloc[0],
            "base": int(int(g.loc[g["p"].idxmax(), "lane"]) == wl),
            "plus": int(int(g.loc[g["p_s"].idxmax(), "lane"]) == wl),
            "lane1": int(wl == 1),
        })
    R = pd.DataFrame(rows)
    R.to_csv("data/scan_confirm_races.csv" if not only else
             "data/scan_confirm_holdout.csv", index=False)

    def rep(sub, label):
        print(f"\n=== {label}  n={len(sub):,}レース / {sub['date'].nunique():,}日 ===")
        d = sub["date"].to_numpy()
        for nm, col in [("1号艇固定(予測不要)", "lane1"), ("既存14特徴", "base"),
                        ("既存+走査", "plus")]:
            print(f"  {nm:20s} 本命的中 {sub[col].mean():.3%}")
        for nm, col in [("既存14特徴", "base"), ("既存+走査", "plus")]:
            lo, hi = cluster_ci(sub[col].to_numpy(float), sub["lane1"].to_numpy(float), d)
            gap = sub[col].mean() - sub["lane1"].mean()
            print(f"  {nm} − 1号艇固定 : {gap * 100:+.2f}pt  "
                  f"CI[{lo * 100:+.2f},{hi * 100:+.2f}]")
        lo, hi = cluster_ci(sub["plus"].to_numpy(float), sub["base"].to_numpy(float), d)
        gap = sub["plus"].mean() - sub["base"].mean()
        verdict = "← 効いとる" if lo > 0 else ("← 悪化" if hi < 0 else "← ゼロを跨いどる")
        print(f"  走査の上乗せ(既存比)  : {gap * 100:+.2f}pt  "
              f"CI[{lo * 100:+.2f},{hi * 100:+.2f}]   {verdict}")

    if only:
        rep(R[R["yr"].isin(only)],
            f"holdout {'/'.join(only)}(走査が一度も見てへん年)")
    else:
        rep(R, "全年")
        rep(R[~R["yr"].isin(SELECTION_YEARS)], "クリーン年のみ(特徴選択に未使用)")


if __name__ == "__main__":
    main()
