"""押し出し検出器を凍結する。前向き記録を始める前の最後の工程。

凍結するもん:
  1. 特徴7本(**朝の出走表だけで作れるもんに限定**。風速・波高はKファイル=レース後なので外した。
     抜いても AUC 0.8518→0.8505 でほとんど落ちんことを確認済み)
  2. 学習データの窓(今日まで)。**以降 再学習せん**
  3. 発火の閾値(上位1%に当たる絶対値)。**以降 動かさん**

保存: data/pushout_model.joblib(モデル+閾値+特徴名+学習窓+指紋)

usage: python -u -m src.pushout_freeze
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import storage
from .boat1_pushed_out import load
from .maegumi_detector import prior_rate

FEATURES = ["grab_max", "grab_sum", "press_max", "press",
            "b1_out", "b1_hold", "v_out"]
TOP_FRAC = 0.01                     # 事前登録: 上位1%
PATH = os.path.join(storage.DATA_DIR, "pushout_model.joblib")


def build_panel():
    df = load()
    df["p_grab"] = prior_rate(df, ["reg", "lane"], "grab1")
    df["p_in"] = prior_rate(df, ["reg", "lane"], "in_")
    df["p_out1"] = prior_rate(df, ["reg"], "out1")
    df["p_hold1"] = prior_rate(df, ["reg"], "hold1")
    df["v_out"] = prior_rate(df, ["jcd"], "out1")
    outer = df[df["lane"] >= 2]
    G = outer.groupby("rid")["p_grab"].agg(grab_max="max", grab_sum="sum")
    P = outer.groupby("rid")["p_in"].agg(press_max="max", press="sum")
    b1 = df[df["lane"] == 1].set_index("rid")
    R = pd.DataFrame(index=b1.index).join(G).join(P)
    R["b1_out"] = b1["p_out1"]
    R["b1_hold"] = b1["p_hold1"]
    R["v_out"] = b1["v_out"]
    R["date"] = b1["date"]
    R["c1"] = b1["course"]
    R["tansho_lane"] = b1["tansho_lane"]
    R = R.reset_index().rename(columns={"index": "rid"})
    R["y"] = (R["c1"] > 1).astype(int)
    return R.dropna(subset=FEATURES).reset_index(drop=True)


def main():
    R = build_panel()
    X = R[FEATURES].to_numpy(float)
    y = R["y"].to_numpy(int)
    print(f"学習: {len(R):,}レース ({R['date'].min()}-{R['date'].max()})  "
          f"押し出し率 {y.mean():.2%}")

    m = CalibratedClassifierCV(HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0,
        random_state=42), method="isotonic", cv=3)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    thr = float(np.quantile(p, 1 - TOP_FRAC))
    print(f"in-sample AUC {roc_auc_score(y, p):.4f}(参考。判定は前向きでやる)")
    print(f"閾値(上位{TOP_FRAC:.0%}) p >= {thr:.6f}")
    fire = p >= thr
    print(f"  発火 {fire.sum():,}レース ({fire.mean():.2%})  "
          f"うち実際に押し出し {y[fire].mean():.2%}  "
          f"1号艇勝率 {(R.loc[fire, 'tansho_lane'] == 1).mean():.2%}")
    days = R["date"].nunique()
    print(f"  → 1日あたり約 {fire.sum() / days:.1f}レース発火"
          f"({days:,}日で {fire.sum():,}本)")

    bundle = {
        "model": m, "features": FEATURES, "threshold": thr,
        "top_frac": TOP_FRAC,
        "train_from": R["date"].min(), "train_to": R["date"].max(),
        "n_train": int(len(R)), "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "note": "押し出し検出器。朝の出走表だけで作る7特徴。再学習も閾値変更もせん。",
    }
    joblib.dump(bundle, PATH)
    h = hashlib.sha256(open(PATH, "rb").read()).hexdigest()
    print(f"\n保存: {PATH}\n  SHA256 {h}")
    meta = {k: v for k, v in bundle.items() if k != "model"}
    meta["sha256"] = h
    with open(os.path.join(storage.DATA_DIR, "pushout_model.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
