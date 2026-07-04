"""上流特徴の洗い出し：生ID(選手登録番号)・未使用軸(気温/水温)が、既存特徴に上乗せsignalを持つか。

問い: 手で選んだ24特徴の外に、データから引ける軸(生reg・temperature・water_temp)が
  OOF弁別を上げるか。permutation importance に「どの軸が効くか」を名指させる。
規律:
  - walk-forward OOF(train<test年)。生IDの過学習は未来年テストで暴かれる(暗記なら上がらん)。
  - regの足切りはtrainのみで算出(leakなし)。出走<MINの選手は "rare"(-1)へ。
  - regの平均的強さは既にnat_win等に入る=生regが足すのは残差/交互作用のみ。
  - フレームワークは全条件LightGBMで固定(既存HistGBMとの差を特徴差と混同せんため)。

例: python -u -m src.test_feature_wash
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .features import FEATURES, build_frame
from .validate import load

NEW_NUM = ["temperature", "water_temp"]
MIN_RACES = 30  # trainでこれ未満の選手は rare にまとめる（足切り）


def hit_rate(df_te, p):
    """本命的中率(レース単位: 予測1位の艇が1着か)。"""
    t = df_te.copy()
    t["p"] = p
    hit = tot = 0
    for _, g in t.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        tot += 1
        hit += int(g.iloc[0]["win"] == 1)
    return hit / tot if tot else 0


def main():
    df = load()
    fr = build_frame(df, impute=False)          # 既存FEATURES(venue_code含む)
    for c in NEW_NUM:
        fr[c] = pd.to_numeric(df[c].values, errors="coerce")
    # 生reg（trainで足切り→カテゴリ符号化）
    reg = pd.to_numeric(df["reg"], errors="coerce").fillna(-1).astype(int).values

    yr = df["yr"].values
    y = df["win"].to_numpy(int)
    tr = np.isin(yr, ["2022", "2023", "2024"])
    te = yr == "2025"
    print(f"train(22-24) {tr.sum():,} / test(25) {te.sum():,}")

    # reg 足切り: train出走数<MIN は rare(-1)
    vc = pd.Series(reg[tr]).value_counts()
    keep = set(vc[vc >= MIN_RACES].index)
    reg_f = np.array([r if r in keep else -1 for r in reg])
    print(f"reg: 全{len(set(reg)):,}選手 / train出走>={MIN_RACES}で採用 {len(keep):,}人 / rare集約 {len(set(reg))-len(keep):,}人\n")

    def make_X(cols, with_reg):
        X = fr[cols].copy()
        cats = ["venue_code"]
        if with_reg:
            X = X.assign(reg=reg_f)
            cats.append("reg")
        for c in cats:
            X[c] = X[c].astype("category")
        return X, cats

    def fit_auc(cols, with_reg):
        X, cats = make_X(cols, with_reg)
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, random_state=42, verbose=-1)
        m.fit(X[tr], y[tr], categorical_feature=cats)
        p = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(y[te], p), hit_rate(df[te], p), m, X

    print("=== OOF弁別: 特徴セット別（LightGBM固定・test=2025）===")
    print(f"{'特徴セット':<28}{'AUC':>8}{'本命的中':>9}")
    a0, h0, *_ = fit_auc(FEATURES, False)
    print(f"{'既存24特徴(base)':<28}{a0:>8.4f}{h0*100:>8.1f}%")
    a1, h1, *_ = fit_auc(FEATURES + NEW_NUM, False)
    print(f"{'+気温/水温':<28}{a1:>8.4f}{h1*100:>8.1f}%  (Δ{ (a1-a0):+.4f})")
    a2, h2, *_ = fit_auc(FEATURES, True)
    print(f"{'+生reg(選手ID)':<28}{a2:>8.4f}{h2*100:>8.1f}%  (Δ{ (a2-a0):+.4f})")
    aF, hF, mF, XF = fit_auc(FEATURES + NEW_NUM, True)
    print(f"{'+全部(気温水温+reg)':<28}{aF:>8.4f}{hF*100:>8.1f}%  (Δ{ (aF-a0):+.4f})")

    print("\n=== permutation importance（全部入りモデル・test2025・AUC低下量）===")
    r = permutation_importance(mF, XF[te], y[te], scoring="roc_auc",
                               n_repeats=5, random_state=42, n_jobs=-1)
    imp = sorted(zip(XF.columns, r.importances_mean, r.importances_std),
                 key=lambda x: -x[1])
    print(f"{'特徴':<16}{'重要度(AUC低下)':>15}{'±std':>9}")
    for name, m, s in imp:
        mark = " ★新規" if name in NEW_NUM + ["reg"] else ""
        print(f"{name:<16}{m:>15.5f}{s:>9.5f}{mark}")
    print("\n★主役はablation(上のΔ)。判定はこっちで下す。")
    print("※perm importanceは【診断用】。perm=0は『無信号』でなく『冗長/既に説明済み』の可能性。")
    print("  これ単独で特徴を落とすな(落とすならablationで確認)。高カードIDはperm依存度が暗記で水増しされる。")
    print("※AUCは6艇全体順位=目的関数でない。張るのはtop-1〜2＝本命的中で見る。")
    print("※未修正の穴: このablationは全レース。bet subset(本命≠1号艇)ではregの残差価値が過小評価されうる。")
    print("※これは弁別の話。回収率(控除率超え)は別途・締切オッズ整備後。")


if __name__ == "__main__":
    main()
