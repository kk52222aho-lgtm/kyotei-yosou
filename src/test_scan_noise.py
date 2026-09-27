"""走査の上乗せ +0.28pt が**情報やのうて列数**やないかを、同じ土俵で偽薬に通す。

## なんで要るか

`test_scan_confirm` は走査24本を載せて **1号艇固定との差が +1.51pt → +1.89pt
(holdout)**、走査の上乗せ **+0.28pt CI[+0.17,+0.39]** を出した。台帳の留保⑤に
そのまま書いてある: **「ノイズ24列を足しても上がらんことは未検証」**。

ここが空いたまま本番に積んだら、**列を24本増やしただけの効果を新発見と読む**。
[[insight_model_residual_is_a_weak_control]] と同じ形や。

## 偽薬の作り方(数字を見る前に決めた)

対照は**日付群と同じ形**にする。列数・尺度・欠損率(62.7%)を揃えんかったら
「弱いから落ちた」筋が通ってまう → [[insight_control_group_design]]。

    shuffle : 走査24本の値を**同じ日の中で行ごとシャッフル**。
              分布も欠損の型もそのままで、**その艇との結びつきだけ壊す**。
              これが本命の対照や(情報が効いとるんかを直接聞く)
    noise   : 同じ欠損の型をかぶせた標準正規24本。留保⑤が名指ししとるやつ。
              木は最初から分岐に使わんので**甘い対照**になる。両方出す

判定は走査と同じ道具(日付クラスタのブートストラップ、同じレース集合、6艇そろい、
確定単勝)。**走査が勝つだけやなく、偽薬2本がゼロを跨ぐこと**を要求する。

usage:
    python -u -m src.test_scan_noise                      # 全年で採点
    python -u -m src.test_scan_noise data/scan_holdout.csv 2025 2026   # holdout
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .scan_all import KEYS, lagged, load as scan_load
from .scan_pit import lagged_pit
from .test_scan_confirm import NBOOT, N_MAX, cluster_ci, pick
from .train import _make_estimator
from .validate import SELECTION_YEARS

RNG = np.random.default_rng(20260927)

# 🚨 --pit を付けると窓を「**その日より前**」にする(scan_pit.lagged_pit)。
#    締切時点で配れる形と学習時の形が一致する版や。ちがいは
#    `test_serve_scan` で測った通りで、そのままやと値の一致が 73.78% しか無い。
PIT = "--pit" in sys.argv
if PIT:
    sys.argv = [a for a in sys.argv if a != "--pit"]
LAG = lagged_pit if PIT else lagged


def make_placebos(F: pd.DataFrame, extra: list[str]) -> dict[str, list[str]]:
    """走査24本と**同じ形**の偽薬を2組つくって列名を返す。"""
    mask = F[extra].isna().to_numpy()          # 欠損の型をそのまま借りる
    out: dict[str, list[str]] = {}

    # ① 日内シャッフル: 分布と欠損率はそのまま、艇との結びつきだけ壊す
    #
    # 🚨 最初の版は **24列を別々に** 並べ替えとった。それやと列をまたいだ欠損の
    #    同時構造が壊れて「24本とも揃っとる行」が **62.8% → 3.0%** に落ちる
    #    (採点の前に形を出す門がこれを捕まえた)。痩せた偽薬が負けても
    #    「情報が無いから」か「形がちがうから」か分かれん
    #    → [[insight_control_group_design]]。
    #    直し: **1日につき並べ替えは1本**。24列の行ベクトルを丸ごと別の艇へ付け替える。
    #    欠損の型も列間の相関もそのままで、**その艇との結びつきだけ**壊れる。
    sh = ["P1_" + c for c in extra]
    V = F[extra].to_numpy(dtype=float).copy()
    day = F["date"].to_numpy()
    order = np.argsort(day, kind="stable")
    bounds = np.flatnonzero(np.r_[True, day[order][1:] != day[order][:-1], True])
    for i in range(len(bounds) - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        V[idx, :] = V[RNG.permutation(idx), :]      # 行ごと丸ごと動かす
    for k, name in enumerate(sh):
        F[name] = V[:, k]
    out["既存+偽薬(日内シャッフル)"] = sh

    # ② ノイズ: 留保⑤が名指ししとるやつ。同じ欠損の型をかぶせる
    nz = []
    for k, c in enumerate(extra):
        name = "P2_" + c
        v = RNG.standard_normal(len(F))
        v[mask[:, k]] = np.nan
        F[name] = v
        nz.append(name)
    out["既存+ノイズ24列"] = nz
    return out


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/scan_all.csv"
    only = sys.argv[2:] or None
    print(("【pit版: 窓はその日より前】" if PIT else "【通常版】")
          + f" 選抜元: {src}" + (f"  / 採点年を {only} に限定" if only else ""))
    res = pd.read_csv(src)
    if not res["survive_ort"].any():
        print("直交化残差の生き残りゼロ。載せるもんが無い。")
        return
    kept = pick(res)

    df = scan_load()
    extra: list[str] = []
    for _, r in kept.iterrows():
        for lname, vals in LAG(df, KEYS[r["key"]], r["qty"]):
            if lname == r["lag"]:
                name = f"S_{r['key']}_{r['qty']}_{r['lag']}"
                df[name] = vals
                extra.append(name)
                break
    assert len(extra) == len(kept), "走査特徴の組み立てが合わん"

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

    arms = {"既存14特徴": FEATURES, "既存+走査": FEATURES + extra}
    for nm, cols in make_placebos(F, extra).items():
        arms[nm] = FEATURES + cols
    keep = F[extra].notna().all(axis=1).mean()
    print(f"\n採点盤: {len(F):,}行 / 走査特徴の非欠測率 {keep:.1%}")
    # 🚨 偽薬が本物と同じ形になっとるか**採点の前に**確かめて、合わんかったら採点せん。
    #    出すだけやと読み飛ばす(実際1回踏んだ: 別々に並べ替えて 62.8%→3.0%)。
    bad = []
    for nm, cols in arms.items():
        add = [c for c in cols if c not in FEATURES]
        k = F[add].notna().all(axis=1).mean() if add else 1.0
        flag = ""
        if add and abs(k - keep) > 0.02:
            flag = "  🚨 本物と形がちがう"
            bad.append((nm, k))
        print(f"  {nm:22s} 列 {len(cols):3d}(追加 {len(add):2d}) "
              f"追加列の非欠測 {k:6.1%}{flag}")
    if bad:
        print(f"\n🚨 偽薬の形が本物({keep:.1%})と 2pt 以上ちがう: "
              + ", ".join(f"{n} {v:.1%}" for n, v in bad)
              + "\n   **痩せた偽薬が負けても情報の話にならんんで採点せん。**")
        return

    preds = {}
    for name, cols in arms.items():
        X = F[cols].to_numpy(dtype=float)
        out = []
        for yv in sorted(F["yr"].unique()):
            te = (F["yr"] == yv).to_numpy()
            tr = (F["yr"].astype(int) < int(yv)).to_numpy()
            if tr.sum() < 3000 or te.sum() < 2000:
                continue
            if only and yv not in only:
                continue   # 採点せん年に当てはめても捨て札(学習集合は年<yv で不変)
            print(f"[fit] {name:22s} {yv} train={tr.sum():,}", flush=True)
            est = _make_estimator()
            est.categorical_features = [cols.index("venue_code")]
            mdl = CalibratedClassifierCV(est, method="isotonic", cv=3)
            mdl.fit(X[tr], y[tr])
            o = F.loc[te, keys + ["tansho_lane", "yr"]].copy()
            o["p"] = mdl.predict_proba(X[te])[:, 1]
            out.append(o)
        preds[name] = pd.concat(out).set_index(keys)

    j = preds["既存14特徴"][["p", "tansho_lane", "yr"]].copy()
    short = {"既存+走査": "scan", "既存+偽薬(日内シャッフル)": "shuf",
             "既存+ノイズ24列": "noise"}
    for nm, sfx in short.items():
        j = j.join(preds[nm][["p"]].rename(columns={"p": "p_" + sfx}), how="inner")
    j = j.reset_index()

    rows = []
    for _, g in j.groupby(["date", "jcd", "rno"], sort=False):
        if len(g) != 6:
            continue
        wl = int(g["tansho_lane"].iloc[0])
        r = {"date": g["date"].iloc[0], "yr": g["yr"].iloc[0],
             "lane1": int(wl == 1),
             "base": int(int(g.loc[g["p"].idxmax(), "lane"]) == wl)}
        for sfx in short.values():
            r[sfx] = int(int(g.loc[g["p_" + sfx].idxmax(), "lane"]) == wl)
        rows.append(r)
    R = pd.DataFrame(rows)
    tag = "_pit" if PIT else ""
    R.to_csv(f"data/scan_noise_holdout{tag}.csv" if only
             else f"data/scan_noise_races{tag}.csv",
             index=False)

    def rep(sub: pd.DataFrame, label: str) -> None:
        if not len(sub):
            print(f"\n=== {label}: 行が無い ===")
            return
        print(f"\n=== {label}  n={len(sub):,}レース / {sub['date'].nunique():,}日 ===")
        d = sub["date"].to_numpy()
        for nm, col in [("1号艇固定(予測不要)", "lane1"), ("既存14特徴", "base"),
                        ("既存+走査", "scan"), ("既存+偽薬(日内シャッフル)", "shuf"),
                        ("既存+ノイズ24列", "noise")]:
            print(f"  {nm:24s} 本命的中 {sub[col].mean():.3%}")
        print("\n  既存14特徴からの上乗せ(同じレース集合・日付クラスタCI):")
        for nm, col in [("走査24本", "scan"), ("偽薬(日内シャッフル)", "shuf"),
                        ("ノイズ24列", "noise")]:
            lo, hi = cluster_ci(sub[col].to_numpy(float), sub["base"].to_numpy(float), d)
            gap = (sub[col].mean() - sub["base"].mean()) * 100
            v = "← 効いとる" if lo > 0 else ("← 悪化" if hi < 0 else "← ゼロを跨ぐ")
            print(f"    {nm:22s} {gap:+.2f}pt  CI[{lo*100:+.2f},{hi*100:+.2f}]  {v}")
        # 🚨 本番の問い: 走査は偽薬より上か。上乗せ同士を直接引く
        for nm, col in [("偽薬(日内シャッフル)", "shuf"), ("ノイズ24列", "noise")]:
            lo, hi = cluster_ci(sub["scan"].to_numpy(float), sub[col].to_numpy(float), d)
            gap = (sub["scan"].mean() - sub[col].mean()) * 100
            v = "← 走査が勝つ" if lo > 0 else ("← 偽薬が勝つ" if hi < 0 else
                                             "🚨 ゼロを跨ぐ=情報と列数が分かれん")
            print(f"    走査 − {nm:18s} {gap:+.2f}pt  "
                  f"CI[{lo*100:+.2f},{hi*100:+.2f}]  {v}")

    if only:
        rep(R[R["yr"].isin(only)], f"holdout {'/'.join(only)}(走査が一度も見てへん年)")
    else:
        rep(R, "全年")
        rep(R[~R["yr"].isin(SELECTION_YEARS)], "クリーン年のみ(特徴選択に未使用)")


if __name__ == "__main__":
    main()
