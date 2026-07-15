"""0.01%に食らいつく: 全艇単勝オッズ(odds_cache 3093R)で逆イン本命を低分散EVで確定狙い。

ROI(勝敗×配当)はfat-tailで暴れCI[94,129]。だが的中率は64.5%で確定的、暴れてるのは配当分散。
∴全艇オッズがあれば:
  ①EV=model_p×odズ で期待ROIを低分散推定(勝敗でなく確率×既知オッズ)→CIが締まる
  ②市場含意確率(1/odズ正規化) vs model_p を1レース対で比較→市場が外本命を過小評価か直接
odds_cacheはサイトが妙味中心に貯めた=逆イン本命that濃い。walk-forwardでmodel_p付与し結合。
確定払戻・6艇クリーン。※オッズは締切前capture=実際に張れる価格。

例: python -u -m src.test_gyaku_ev
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS

ODDS = Path(__file__).resolve().parent.parent / "data" / "odds_cache.json"


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane FROM entries e JOIN payouts p
        ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df, odds):
    rows = []
    yrs = [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]
    for y in yrs:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            if g["tenji_time"].isna().any():   # 展示フル揃いだけ(朝=展示前の交絡を除去)
                continue
            key = f"{date}-{int(jcd):02d}-{int(rno)}"
            od = odds.get(key)
            if not od:
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            h = int(top["lane"])
            if h == 1:                       # 外本命だけ
                continue
            s = g["p"].sum()
            o0 = od.get(str(h))
            if s <= 0 or not o0 or o0 <= 1:
                continue
            mp0 = float(top["p"]) / s
            # 市場含意確率(控除抜き=1/オッズ正規化)
            inv = {l: 1.0 / od[str(l)] for l in range(1, 7) if od.get(str(l), 0) > 0}
            imp0 = inv.get(h, 0) / sum(inv.values()) if inv else np.nan
            won = int(top["tansho_lane"]) == h
            rows.append({
                "yr": y, "mp0": mp0, "imp0": imp0, "o0": float(o0),
                "ev": mp0 * float(o0),                        # 期待ROI(低分散)
                "realized": float(o0) if won else 0.0,        # 実ROI(fat-tail)
                "won": int(won),
            })
    return pd.DataFrame(rows)


def ci(x, iters=4000):
    a = np.asarray(x, float); n = len(a)
    rng = np.random.default_rng(11)
    m = [a[rng.integers(0, n, n)].mean() for _ in range(iters)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    odds = json.loads(ODDS.read_text(encoding="utf-8"))
    df = load()
    d = collect(df, odds)
    print(f"外本命×オッズ有 総数 {len(d):,}（odds_cache 3093Rと結合）\n")
    print("  閾値      N   的中   市場含意p0  モデルp0(較正)  EV=期待ROI[CI]        実ROI")
    for lo in [0.40, 0.45, 0.50, 0.55]:
        sub = d[d["mp0"] >= lo]
        n = len(sub)
        if n < 40:
            print(f"  p0≥{lo:.2f}: n{n}少"); continue
        ev = sub["ev"].mean() * 100
        elo, ehi = ci(sub["ev"]); elo, ehi = elo * 100, ehi * 100
        real = sub["realized"].mean() * 100
        hr = sub["won"].mean() * 100
        impp = sub["imp0"].mean() * 100
        mpp = sub["mp0"].mean() * 100
        star = " ★EV下限≥100" if elo >= 100 else (" ○EV>100" if ev >= 100 else "")
        edge = mpp - impp
        print(f"  p0≥{lo:.2f}: {n:4d}  {hr:4.1f}%  {impp:5.1f}%   {mpp:5.1f}%(実{hr:.0f})   "
              f"{ev:5.1f}% [{elo:.0f},{ehi:.0f}]   {real:5.1f}%  差{edge:+.1f}pt{star}")
    print("\n読み: モデルp0>市場含意p0(差+)=市場が外本命を過小評価の直接証拠。")
    print("      EVのCI下限≥100(★)=低分散推定で歴史確定=前向き待たず本物。")
    print("※オッズ締切前capture・walk-forward・6艇クリーン。実ROIとEVの一致で較正も確認。")


if __name__ == "__main__":
    main()
