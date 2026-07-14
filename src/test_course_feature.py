"""選手コース適性(個性)を"モデル特徴"として足したら上がるか＝個性の真価テスト。

コース適性は受益者選抜で最強信号(90%)を見せた=本物。但し単勝では市場に織込まれ届かず。
∴選抜でなく特徴量として: reg×lane の勝率(apt_w)/2連率(apt2) をモデルに足し、
ベース vs +個性 で 1着精度 / 妙味(本命≠1号)単勝ROI の上乗せを測る。上がれば初の壁破り。
確定払戻・walk-forward・リークなし。

例: python -u -m src.test_course_feature
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator
from .validate import SELECTION_YEARS

CANDS = ["apt_w", "apt2"]
MIN_APT = 5


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def add_apt(df):
    f = df["finish"].fillna(99)
    df["_w"] = (f == 1).astype(float)
    df["_2"] = ((f >= 1) & (f <= 2)).astype(float)
    df = df.sort_values(["reg", "lane", "date", "rno"]).reset_index(drop=True)
    g = df.groupby(["reg", "lane"], sort=False)
    nb = g.cumcount().values
    for src, dst in [("_w", "apt_w"), ("_2", "apt2")]:
        before = (g[src].cumsum() - df[src]).values
        df[dst] = np.where(nb >= MIN_APT, before / np.clip(nb, 1, None), np.nan)
    return df


def fit_eval(df, extra):
    cols = FEATURES + extra
    X = pd.concat([build_frame(df, impute=False)[FEATURES].reset_index(drop=True),
                   df[CANDS].reset_index(drop=True)], axis=1)[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    acc, rn, rd = [], 0.0, 0
    for yv in [v for v in sorted(df["yr"].unique()) if v not in SELECTION_YEARS]:
        te = (df["yr"] == yv).to_numpy()
        tr = (df["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        out = df[te].copy()
        out["p"] = m.predict_proba(X[te])[:, 1]
        for _, g in out.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if pd.isna(top["tansho_lane"]):
                continue
            acc.append(int(top["tansho_lane"]) == int(top["lane"]))
            if int(top["lane"]) != 1:
                rd += 1
                if int(top["tansho_lane"]) == int(top["lane"]):
                    rn += float(top["tansho_yen"])
    return np.mean(acc) * 100, (rn / (rd * 100) * 100 if rd else 0)


def main():
    df = load()
    df = add_apt(df)
    print(f"コース適性 充足 apt_w {df['apt_w'].notna().mean()*100:.0f}% / apt2 {df['apt2'].notna().mean()*100:.0f}%\n")
    base = fit_eval(df, [])
    print(f"  ベース(14特徴)   : 1着{base[0]:.1f}% / 妙味単勝{base[1]:.0f}%")
    plus = fit_eval(df, CANDS)
    print(f"  +コース適性(個性): 1着{plus[0]:.1f}% / 妙味単勝{plus[1]:.0f}%")
    print(f"\n  -> 上乗せ: 1着 {plus[0]-base[0]:+.1f}pt / 妙味単勝 {plus[1]-base[1]:+.0f}pt")
    if plus[0] - base[0] >= 0.5 or plus[1] - base[1] >= 5:
        print("  ** 上がった！ 個性that初の壁破り＝本物か個別検証へ。")
    else:
        print("  個性もモデル特徴では上乗せ無し＝市場効率で天井、で確定（選抜での+4ptも控除に消える）。")
    print("\n※確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
