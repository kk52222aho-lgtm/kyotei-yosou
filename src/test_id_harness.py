"""生ID harness: {reg, motor, boat} が1着signalに上乗せするか、生カテゴリ vs ターゲット符号化で。

判定物差し(お前の指示):
  - bet-subset(本命≠1号艇)の1着的中 が主。AUCは従(目的関数でない)。
  - 生カテゴリ=交互作用(選手×条件の残差)を拾う本命。walk-forwardで暗記は暴かれる。
  - ターゲット符号化(prior年のみ)=marginalの冗長チェック。regのTEはnat_win再生産の想定。
  - motor/boatは節で再割当=venue-scoped((場,NO)複合)で符号化。でないとreg以上に暗記。
  - perm importanceは使わん(診断格下げ済)。判定はablation Δのみ。
  - 多年walk-forward(train<Y / test=Y, Y=23,24,25)をプール。

例: python -u -m src.test_id_harness
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from . import storage
from .features import FEATURES, build_frame
from .validate import load

MB = os.path.join(storage.DATA_DIR, "motor_boat.csv")
MIN_ID = 30       # 生カテゴリ: train出走<この数のIDはrare(-1)へ
ALPHA = 20        # ターゲット符号化の平滑化


def lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=200, random_state=42, verbose=-1)


def te_map(tr_df, keys, alpha=ALPHA):
    """leak-freeターゲット符号化: keyごとの1着率(平滑化)。gmはデフォルト。"""
    gm = tr_df["win"].mean()
    s = tr_df.groupby(keys)["win"].agg(["sum", "count"])
    enc = (s["sum"] + alpha * gm) / (s["count"] + alpha)
    return enc, gm


def hitrate(df_te, p, contrarian_only):
    t = df_te.copy(); t["p"] = p
    hit = tot = 0
    for _, g in t.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        if contrarian_only and int(g.iloc[0]["lane"]) == 1:
            continue
        tot += 1
        hit += int(g.iloc[0]["win"] == 1)
    return hit, tot


def main():
    df = load()
    if os.path.exists(MB):
        mb = pd.read_csv(MB, dtype={"date": str, "jcd": str})
        df = df.merge(mb, on=["date", "jcd", "rno", "lane"], how="left")
        print(f"motor/boat 結合: motor_no欠損 {df.motor_no.isna().mean()*100:.1f}%")
    else:
        df["motor_no"] = np.nan; df["boat_no"] = np.nan
        print("motor_boat.csv 未生成→reg のみ評価")

    fr = build_frame(df, impute=False)
    reg = pd.to_numeric(df["reg"], errors="coerce").fillna(-1).astype(int).values
    # venue-scoped 複合キー（場*1000+NO）
    mkey = (pd.to_numeric(df["jcd"], errors="coerce").fillna(0) * 1000
            + pd.to_numeric(df["motor_no"], errors="coerce").fillna(-1)).astype(int).values
    bkey = (pd.to_numeric(df["jcd"], errors="coerce").fillna(0) * 1000
            + pd.to_numeric(df["boat_no"], errors="coerce").fillna(-1)).astype(int).values
    yr = df["yr"].values
    y = df["win"].to_numpy(int)
    win_s = df["win"]

    # 評価対象の設定リスト: (名前, 追加特徴の作り方, カテゴリか)
    def build_raw(idarr, tr):
        vc = pd.Series(idarr[tr]).value_counts()
        keep = set(vc[vc >= MIN_ID].index)
        return np.array([v if v in keep else -1 for v in idarr]), len(keep)

    configs = ["base",
               "reg_raw", "reg_TE",
               "motor_raw", "motor_TE",
               "boat_raw", "boat_TE"]

    # 多年walk-forwardでtest予測を貯める
    preds = {c: [] for c in configs}
    idx_te_all = []
    for Y in ["2023", "2024", "2025"]:
        tr = np.isin(yr, ["2022", "2023", "2024", "2025"]) & (yr < Y)
        te = yr == Y
        idx_te_all.append(np.where(te)[0])
        Xb = fr[FEATURES].copy()
        Xb["venue_code"] = Xb["venue_code"].astype("category")

        def fit_pred(extra_cols, cats):
            X = Xb.copy()
            for c, v in extra_cols.items():
                X[c] = v
            for c in cats:
                X[c] = X[c].astype("category")
            m = lgbm(); m.fit(X[tr], y[tr], categorical_feature=["venue_code"] + cats)
            return m.predict_proba(X[te])[:, 1]

        # base
        preds["base"].append(fit_pred({}, []))
        # reg
        reg_f, _ = build_raw(reg, tr)
        preds["reg_raw"].append(fit_pred({"reg": reg_f}, ["reg"]))
        enc, gm = te_map(df[tr], "reg"); reg_te = df["reg"].map(enc).fillna(gm).values
        preds["reg_TE"].append(fit_pred({"reg_te": reg_te}, []))
        # motor (venue-scoped)
        mk_f, _ = build_raw(mkey, tr)
        preds["motor_raw"].append(fit_pred({"mkey": mk_f}, ["mkey"]))
        dfm = df.assign(_mk=mkey); enc, gm = te_map(dfm[tr], "_mk"); mte = pd.Series(mkey).map(enc).fillna(gm).values
        preds["motor_TE"].append(fit_pred({"mte": mte}, []))
        # boat (venue-scoped)
        bk_f, _ = build_raw(bkey, tr)
        preds["boat_raw"].append(fit_pred({"bkey": bk_f}, ["bkey"]))
        dfb = df.assign(_bk=bkey); enc, gm = te_map(dfb[tr], "_bk"); bte = pd.Series(bkey).map(enc).fillna(gm).values
        preds["boat_TE"].append(fit_pred({"bte": bte}, []))
        print(f"  {Y} 完了")

    te_idx = np.concatenate(idx_te_all)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    print("\n=== bet-subset(本命≠1号艇) 1着的中 が主。多年プール(test23-25) ===")
    print(f"{'config':<11}{'bet-subset的中':>14}{'(N)':>8}{'全レース的中':>12}{'AUC':>8}{'Δbet(vs base)':>14}")
    base_bet = None
    for c in configs:
        p = np.concatenate(preds[c])
        h, n = hitrate(df_te, p, True)
        ha, na = hitrate(df_te, p, False)
        auc = roc_auc_score(df_te["win"].to_numpy(int), p)
        br = h / n
        if c == "base":
            base_bet = br
        d = (br - base_bet) * 100
        print(f"{c:<11}{br*100:>13.2f}%{n:>8,}{ha/na*100:>11.2f}%{auc:>8.4f}{d:>+13.2f}")
    print("\n※主=bet-subset 1着的中Δ。生rawが>0で TEが≒0なら『交互作用に信号・marginalは冗長』。")
    print("※両方≒0ならそのIDは1着signalに無寄与(rateに吸収済)。生rawだけ大きく負ならvenue跨ぎ暗記。")
    print("※これは1着弁別の増強。回収率(市場超え)は最強1着モデルができてから締切オッズで審判。")


if __name__ == "__main__":
    main()
