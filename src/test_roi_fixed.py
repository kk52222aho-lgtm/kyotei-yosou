"""固定universe ROI-permutation: selection効果を消して per-bet品質の本体を測る。

free版(test_roi_importance)はbet母集団が特徴shuffleで激変し、ROI-Δが「賭けを選ぶ効果」に
支配された。ここではbet母集団=baseモデルのcontrarian 3,675レースに【固定】。その固定集合の中で
特徴をshuffle→本命(argmax)/2連単top-3を選び直し→ROI。件数固定=ROI-Δは純per-bet品質。

判定:
  - 固定universeでどの特徴をshuffleしてもROIが崩れない → 儲けは純粋にselection(戦略構造)、
    特徴はper-bet品質の本体でない。魚と違い信号は特徴でなく「本命≠1号艇を張る構造」に宿る。
  - 特定特徴のshuffleでROIが大きく崩れる → それがper-bet品質の本体。
リーク釘: train無傷/test-onlyshuffle。母集団はbase(無shuffle)予測で固定。

例: python -u -m src.test_roi_fixed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .features import FEATURES, build_frame
from .strategy_search import exacta_probs
from .validate import load, _make_estimator

YEARS = ["2023", "2024", "2025"]
N_REPEAT = 20


def race_structs(df_te):
    d = df_te.reset_index(drop=True)
    st = []
    for _, g in d.groupby(["date", "jcd", "rno"]):
        r0 = g.iloc[0]
        ey = r0["exacta_yen"]
        st.append({
            "idx": g.index.to_numpy(),
            "lanes": g["lane"].to_numpy(int),
            "tansho_lane": int(r0["tansho_lane"]) if not pd.isna(r0["tansho_lane"]) else -1,
            "tansho_yen": int(r0["tansho_yen"]) if not pd.isna(r0["tansho_yen"]) else 0,
            "exacta_combo": r0["exacta_combo"],
            "exacta_yen": None if pd.isna(ey) else int(ey),
            "base_contrarian": False,
        })
    return st


def roi_fixed(p, structs):
    """固定集合(base_contrarian=True)のみ。本命=argmax(p)を張る(lane1でも張る=件数固定)。"""
    tr = ts = er = es = bet = 0
    for s in structs:
        if not s["base_contrarian"]:
            continue
        pr = p[s["idx"]]
        o = np.argsort(pr)[::-1]
        top1 = s["lanes"][o[0]]
        bet += 1
        ts += 100
        if s["tansho_lane"] == top1:
            tr += s["tansho_yen"]
        if s["exacta_yen"] is not None:
            ssum = pr.sum()
            if ssum > 0:
                pdd = {int(l): float(v) / ssum for l, v in zip(s["lanes"], pr)}
                top3 = [c for c, _ in exacta_probs(pdd)[:3]]
                es += 300
                if s["exacta_combo"] in top3:
                    er += s["exacta_yen"]
    return np.array([tr, ts, er, es, bet], float)


def main():
    df = load()
    fr = build_frame(df, impute=False)[FEATURES].to_numpy(float)
    y = df["win"].to_numpy(int)
    yr = df["yr"].values

    cache = []
    for Y in YEARS:
        tr = yr < Y; te = yr == Y
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(fr[tr], y[tr])
        Xte = fr[te].copy()
        st = race_structs(df[te])
        pbase = m.predict_proba(Xte)[:, 1]
        # base(無shuffle)予測で contrarian 母集団を固定
        for s in st:
            pr = pbase[s["idx"]]
            if s["lanes"][np.argmax(pr)] != 1:
                s["base_contrarian"] = True
        cache.append((m, Xte, st))
        print(f"  {Y} 学習完了 (固定bet {sum(s['base_contrarian'] for s in st):,})")

    def total(perm_feat=None, seed=0):
        acc = np.zeros(5)
        for (m, Xte, st) in cache:
            X = Xte
            if perm_feat is not None:
                X = Xte.copy()
                rng = np.random.default_rng(seed)
                X[:, perm_feat] = X[rng.permutation(len(X)), perm_feat]
            p = m.predict_proba(X)[:, 1]
            acc += roi_fixed(p, st)
        return acc

    def rois(a):
        return (a[0] / a[1] if a[1] else 0, a[2] / a[3] if a[3] else 0,
                (a[0] + a[2]) / (a[1] + a[3]) if (a[1] + a[3]) else 0, int(a[4]))

    base = total()
    bt, be, bc, bbet = rois(base)
    print("\n" + "=" * 66)
    print(f"BASELINE (固定universe={bbet:,}レース・base picksそのまま)")
    print(f"  単勝 {bt*100:.1f}% / 2連単 {be*100:.1f}% / 合計 {bc*100:.1f}%")
    print("=" * 66)

    print(f"\n=== 固定universe ROI-permutation ({N_REPEAT}回・中央値Δ・件数固定) ===")
    print(f"{'特徴':<14}{'単勝Δ':>9}{'2連単Δ':>9}{'合計Δ':>9}{'跨0?':>10}")
    res = []
    for f, name in enumerate(FEATURES):
        dc = []; dt = []; de = []
        for k in range(N_REPEAT):
            t, e, c, _ = rois(total(perm_feat=f, seed=k))
            dt.append((t - bt) * 100); de.append((e - be) * 100); dc.append((c - bc) * 100)
        lo, hi = np.percentile(dc, [25, 75])
        cross = "効く" if hi < 0 else ("逆/無" if lo > 0 else "跨ぐ")
        res.append((name, np.median(dt), np.median(de), np.median(dc), cross))
    for name, mt, me, mc, cross in sorted(res, key=lambda x: x[3]):
        print(f"{name:<14}{mt:>+8.1f}{me:>+9.1f}{mc:>+8.1f}{cross:>11}")
    print("\n※件数固定=ROI-Δは純per-bet品質。大きく負=その特徴がper-betの本体。")
    print("※全特徴で≒0 → 儲けは特徴でなく戦略構造(本命≠1号艇のselection)に宿る=魚と質が違う。")


if __name__ == "__main__":
    main()
