"""#1(場/venue_code)を抜いたら残りは効くか＝場that全部やってるのか他も効くのかを割る。

場のcategorical化that AUC0.72→0.83のブレイクスルー。∴問い: 場を抜くとモデルは崩れるか、
残り13特徴で立つか。leave-one-out+単独で1着精度を測り、各特徴の真の限界貢献を出す。
1着精度=モデルtop pick==実単勝当たりの率。1号ベタ買い≈53.6%that下限ベンチ。確定払戻・walk-forward。

例: python -u -m src.test_ablation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from . import storage
from .features import FEATURES, build_frame
from .validate import SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def est(cat_idx):
    cats = [cat_idx] if cat_idx is not None else []
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, categorical_features=cats, random_state=42)


def acc_for(full, y, df, cols):
    """cols(特徴名のsubset)で walk-forward 1着精度。"""
    cat_idx = cols.index("venue_code") if "venue_code" in cols else None
    X = full[cols].to_numpy(dtype=float)
    accs = []
    for yv in [v for v in sorted(df["yr"].unique()) if v not in SELECTION_YEARS]:
        te = (df["yr"] == yv).to_numpy()
        tr = (df["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        m = CalibratedClassifierCV(est(cat_idx), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        out = df[te].copy()
        out["p"] = m.predict_proba(X[te])[:, 1]
        for _, g in out.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            accs.append(int(top["tansho_lane"]) == int(top["lane"]))
    return np.mean(accs) * 100


def main():
    df = load()
    full = build_frame(df, impute=False)[FEATURES].reset_index(drop=True)
    y = df["win"].to_numpy(dtype=int)
    rest = [c for c in FEATURES if c not in ("venue_code", "lane")]

    configs = [
        ("全14特徴", FEATURES),
        ("－場(venue)", [c for c in FEATURES if c != "venue_code"]),
        ("－艇番(lane)", [c for c in FEATURES if c != "lane"]),
        ("場+艇番だけ(2)", ["venue_code", "lane"]),
        ("場だけ(1)", ["venue_code"]),
        ("艇番だけ(1)", ["lane"]),
        ("場も艇番も抜く(残12)", rest),
    ]
    print("1着精度(walk-forward・確定払戻)  ※1号ベタ買い≈53.6%that下限\n")
    base = None
    for name, cols in configs:
        a = acc_for(full, y, df, cols)
        if name == "全14特徴":
            base = a
            print(f"  {name:22s}: {a:.1f}%")
        else:
            print(f"  {name:22s}: {a:.1f}%   (全14比 {a-base:+.1f}pt)")
    print("\n読み: 『場+艇番だけ』that全14に肉薄なら他12は飾り。『－場』that大きく落ちれば場that柱。")
    print("      『残12(場も艇番も無し)』thaが53.6%を超えれば選手/機力/展示も独立に効いてる。")


if __name__ == "__main__":
    main()
