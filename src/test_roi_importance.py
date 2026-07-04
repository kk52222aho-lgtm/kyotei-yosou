"""ROI-permutation: どの変数が「精度」でなく「儲け(戦略ROI)」を生んどるか。

これまでのpermutationは全部AUC/1着的中=精度ターゲット。低オッズ帯で薄く削る戦略では
精度に効く変数と儲けに効く変数がズレる(②で実証)。そこで:
  ターゲット=戦略ROI(単勝+2連単・確定払戻ベース=土台116/188と同じ土俵)。
  test年でtest-onlyに特徴を1個shuffle→再予測→本命≠1号艇のbet再導出→ROI崩れ幅。

4つの穴埋め(お前の指示):
  1. n=REPEAT回shuffleして崩れ幅の分布(中央値・ゼロ跨ぎ)。単発は乱数のアタリハズレ。
  2. bet件数も一緒に吐く。ROI崩れが「質低下」か「賭け数激減」か分離。
  3. 単勝/2連単は同一shuffle系列(同一pp)から導出=差は戦略構造の差と言い切れる。
  4. baseline ROI(shuffleなし)を出力ヘッダに焼き込む。崩れ幅はbaseline比でしか意味持たん。
リーク釘: train年は無傷、test年だけshuffle。in-sample損益permutationはリークで全部効く嘘。

例: python -u -m src.test_roi_importance
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
    """test年を1レース1構造体に。位置indexはp配列(=X[te]順)に整合。"""
    d = df_te.reset_index(drop=True)
    structs = []
    for _, g in d.groupby(["date", "jcd", "rno"]):
        idx = g.index.to_numpy()
        r0 = g.iloc[0]
        ey = r0["exacta_yen"]
        structs.append({
            "idx": idx,
            "lanes": g["lane"].to_numpy(int),
            "tansho_lane": int(r0["tansho_lane"]) if not pd.isna(r0["tansho_lane"]) else -1,
            "tansho_yen": int(r0["tansho_yen"]) if not pd.isna(r0["tansho_yen"]) else 0,
            "exacta_combo": r0["exacta_combo"],
            "exacta_yen": None if pd.isna(ey) else int(ey),
        })
    return structs


def roi_parts(p, structs):
    """戦略ROIの構成部品(returns/stakes/bet件数)を返す。単勝と2連単は同一pから。"""
    tr = ts = er = es = bet = 0
    for s in structs:
        pr = p[s["idx"]]
        o = np.argsort(pr)[::-1]
        top1 = s["lanes"][o[0]]
        if top1 == 1:                 # 本命=1号艇は見送り(contrarian戦略)
            continue
        bet += 1
        ts += 100
        if s["tansho_lane"] == top1:
            tr += s["tansho_yen"]
        if s["exacta_yen"] is not None:
            ssum = pr.sum()
            if ssum > 0:
                pd_ = {int(l): float(v) / ssum for l, v in zip(s["lanes"], pr)}
                top3 = [c for c, _ in exacta_probs(pd_)[:3]]
                es += 300
                if s["exacta_combo"] in top3:
                    er += s["exacta_yen"]
    return np.array([tr, ts, er, es, bet], float)


def main():
    df = load()
    fr = build_frame(df, impute=False)[FEATURES].to_numpy(float)
    y = df["win"].to_numpy(int)
    yr = df["yr"].values

    # 年ごと: モデル学習(無傷)＋test構造体をキャッシュ
    cache = []
    for Y in YEARS:
        tr = yr < Y
        te = yr == Y
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(fr[tr], y[tr])
        Xte = fr[te].copy()
        st = race_structs(df[te])
        cache.append((m, Xte, st))
        print(f"  {Y} 学習完了 (test {te.sum():,})")

    def total_parts(perm_feat=None, seed=0):
        acc = np.zeros(5)
        for (m, Xte, st) in cache:
            X = Xte
            if perm_feat is not None:
                X = Xte.copy()
                rng = np.random.default_rng(seed)
                X[:, perm_feat] = X[rng.permutation(len(X)), perm_feat]
            p = m.predict_proba(X)[:, 1]
            acc += roi_parts(p, st)
        return acc  # [tan_ret, tan_stake, ex_ret, ex_stake, bet]

    def rois(a):
        tan = a[0] / a[1] if a[1] else 0
        ex = a[2] / a[3] if a[3] else 0
        comb = (a[0] + a[2]) / (a[1] + a[3]) if (a[1] + a[3]) else 0
        return tan, ex, comb, int(a[4])

    base = total_parts()
    bt, be, bc, bbet = rois(base)
    print("\n" + "=" * 68)
    print(f"BASELINE ROI (shuffleなし・walk-forward通算 test{'/'.join(YEARS)})")
    print(f"  単勝 {bt*100:.1f}% / 2連単3点 {be*100:.1f}% / 合計 {bc*100:.1f}% / bet {bbet:,}レース")
    print("=" * 68)

    print(f"\n=== ROI-permutation (各特徴{N_REPEAT}回shuffle・中央値Δ・baseline比) ===")
    print(f"{'特徴':<14}{'単勝Δ':>9}{'2連単Δ':>9}{'合計Δ':>9}{'bet数Δ':>9}{'合計Δ跨0?':>10}")
    results = []
    for f, name in enumerate(FEATURES):
        dtan = []; dex = []; dcomb = []; dbet = []
        for k in range(N_REPEAT):
            a = total_parts(perm_feat=f, seed=k)
            t, e, c, bet = rois(a)
            dtan.append((t - bt) * 100); dex.append((e - be) * 100)
            dcomb.append((c - bc) * 100); dbet.append(bet - bbet)
        mc = np.median(dcomb)
        lo, hi = np.percentile(dcomb, [25, 75])
        cross = "跨ぐ(不確)" if lo < 0 < hi else ("効く" if hi < 0 else "無/逆")
        results.append((name, np.median(dtan), np.median(dex), mc, np.median(dbet), cross))
    # 合計Δの小さい順(=shuffleでROI一番崩れる=効いとる)に
    for name, mt, me, mc, mb, cross in sorted(results, key=lambda x: x[3]):
        print(f"{name:<14}{mt:>+8.1f}{me:>+9.1f}{mc:>+8.1f}{mb:>+9.0f}{cross:>12}")
    print("\n※合計Δが負に大きい=shuffleでROI崩れる=その変数が儲けの本体。")
    print("※Δが正=shuffleでROI上がる=その変数は儲けにマイナス(精度に効いても本命≠1号艇戦略には逆風)。")
    print("※bet数Δがデカい=shuffleで賭ける母数自体が動く=ROI崩れの意味が『質』でなく『件数』。")
    print("※跨0=25-75%tileがゼロを跨ぐ=分散でかく判定保留(単発ノイズ掴み回避)。")


if __name__ == "__main__":
    main()
