"""B: 市場が取りこぼす断面を"監査付き"で探す。単勝ROI>100の小断面が在るか。

今日110.9%を殺したのと同じ多重比較の罠を最初から潰す設計:
 ①セル網を事前登録して K を数える(多重比較補正の分母) ②walk-forward(train<年→test=年・
   SELECTION_YEARS除外) ③各セル N・年別ROI・racer_id(reg)クラスタbootstrap CI ④point>100の
   候補だけCI算出→naive95%とBonferroni(1-.05/K)の両方で"生き残り"を判定。期待順位:偶然の
   生存(0.05K個)を超える本物が在るか。無ければ「K断面探して控除超え無し」=市場効率の確定。

例: python -u -m src.test_cross_section_search
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS

CONF_GYAKU = [50, 55, 60, 65]     # 逆イン本命(top≠1号)の確信閾
CONF_TOP = [60, 65, 70]           # 本命(top pick)を素直に買う確信閾
CLASSES = ["A1", "A2", "B1", "B2"]


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen FROM entries e JOIN payouts p
        ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df):
    """各テスト年でモデルを張り、レースごとに本命(top pick)1点の結果を集める。"""
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for _, g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            s = g["p"].sum()
            if s <= 0 or pd.isna(top["tansho_lane"]):
                continue
            won = int(top["tansho_lane"]) == int(top["lane"])
            rows.append({
                "yr": y, "jcd": str(top["jcd"]), "reg": str(top["reg"]),
                "cls": str(top["racer_class"]), "conf": float(top["p"]) / s * 100,
                "gyaku": int(int(top["lane"]) != 1),
                "won": int(won),
                "ret": float(top["tansho_yen"]) if won else 0.0,
            })
    return pd.DataFrame(rows)


def build_cells(venues):
    """事前登録セル網。返り: [(kind, conf, seg_type, seg_val, label)]。"""
    cells = []
    for c in CONF_GYAKU:
        cells.append(("gyaku", c, "all", None, f"逆イン conf≥{c} 全体"))
        for v in venues:
            cells.append(("gyaku", c, "venue", v, f"逆イン conf≥{c} 場{v}"))
        for cl in CLASSES:
            cells.append(("gyaku", c, "cls", cl, f"逆イン conf≥{c} {cl}"))
    for c in CONF_TOP:
        cells.append(("top", c, "all", None, f"本命 conf≥{c} 全体"))
        for v in venues:
            cells.append(("top", c, "venue", v, f"本命 conf≥{c} 場{v}"))
    return cells


def subset(d, kind, conf, seg_type, seg_val):
    m = d["conf"] >= conf
    if kind == "gyaku":
        m &= d["gyaku"] == 1
    if seg_type == "venue":
        m &= d["jcd"] == seg_val
    elif seg_type == "cls":
        m &= d["cls"] == seg_val
    return d[m]


def cluster_ci(sub, iters, pct):
    """reg(選手)クラスタbootstrapで単勝ROIのCI。選手ごと全ベットを束で再標本。"""
    regs = sub["reg"].to_numpy()
    ret = sub["ret"].to_numpy()
    uniq = pd.unique(regs)
    idx_by = {r: np.where(regs == r)[0] for r in uniq}
    rng = np.random.default_rng(7)
    rois = []
    for _ in range(iters):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        sel = np.concatenate([idx_by[r] for r in pick])
        rois.append(ret[sel].sum() / (len(sel) * 100) * 100)
    lo = (100 - pct) / 2
    return np.percentile(rois, lo), np.percentile(rois, 100 - lo)


def main():
    df = load()
    d = collect(df)
    venues = sorted(d["jcd"].unique())
    cells = build_cells(venues)
    K = len(cells)
    print(f"収集 本命1点 N={len(d):,}  対象年{sorted(d['yr'].unique())}  選手数{d['reg'].nunique():,}")
    print(f"事前登録セル数 K={K}  → 多重比較: 偶然のnaive95%生存の期待≈{0.05*K:.1f}個 / "
          f"Bonferroni閾=1-0.05/{K}={1-0.05/K:.4f}\n")

    results = []
    for kind, conf, st, sv, label in cells:
        sub = subset(d, kind, conf, st, sv)
        n = len(sub)
        if n < 100:
            continue
        roi = sub["ret"].sum() / (n * 100) * 100
        yrs = {y: (sub[sub.yr == y]["ret"].sum() /
                   (max(len(sub[sub.yr == y]), 1) * 100) * 100) for y in sorted(d.yr.unique())}
        all_yr_pos = all(v > 100 for v in yrs.values())
        results.append({"label": label, "n": n, "roi": roi, "yrs": yrs,
                        "all_yr_pos": all_yr_pos, "sub": sub})

    results.sort(key=lambda x: -x["roi"])
    print(f"評価できたセル {len(results)}（N≥100）。ROI上位20:")
    print("  ROI%   N     年別(全年>100?)          セル")
    for r in results[:20]:
        ys = " ".join(f"{y[2:]}:{v:.0f}" for y, v in r["yrs"].items())
        flag = " ★全年>100" if r["all_yr_pos"] else ""
        print(f"  {r['roi']:5.1f}  {r['n']:5d}  {ys}{flag}   {r['label']}")

    cand = [r for r in results if r["roi"] > 100]
    print(f"\npoint ROI>100 の候補 {len(cand)}個（偶然期待≈{0.05*K:.1f})。CIで検品:")
    naive_surv, bonf_surv = [], []
    for r in cand:
        lo95, hi95 = cluster_ci(r["sub"], 1500, 95)
        loB, hiB = cluster_ci(r["sub"], 1500, 100 * (1 - 0.05 / K))
        tag = []
        if lo95 > 100:
            naive_surv.append(r["label"]); tag.append("naive95下限>100")
        if loB > 100:
            bonf_surv.append(r["label"]); tag.append("★Bonferroni生存")
        print(f"  {r['label']}: ROI{r['roi']:.1f} N{r['n']} "
              f"クラスタCI95[{lo95:.0f},{hi95:.0f}] Bonf[{loB:.0f},{hiB:.0f}] "
              f"{'/'.join(tag) if tag else '—(breakevenと区別不可)'}")

    print(f"\n=== 判定 ===")
    print(f"naive95%下限>100: {len(naive_surv)}個(偶然期待{0.05*K:.1f}) / "
          f"Bonferroni生存(多重比較補正後): {len(bonf_surv)}個")
    if bonf_surv:
        print("Bonferroni生存セル→前向きprobe候補:", bonf_surv)
    else:
        print(f"→ K={K}断面を探して、多重比較補正後に控除超えは【無し】= 単勝は市場効率。")
    print("※単勝のみ。オッズ時系列が無いので締切前ミスプライスは別問題。probeは触らず継続。")


if __name__ == "__main__":
    main()
