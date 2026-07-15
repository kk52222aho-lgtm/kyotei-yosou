"""3連単は何点買えば過去分は平均で当たってたか＝top-N点の的中率(カバー率)を出す。

既存test_trifecta_pointsは妙味(本命≠1号)限定=今日崩れたsurvivorship断面。∴クリーン
(6艇そろい・全レース)で測り直す。モデルのHarville確率でtop-N点買った時、実3連単that
その中に入る率(=的中率)とROIをNごとに。全体+堅い(本命=1号)+妙味(本命≠1号)で。
「平均で当たる」=的中率50%を超えるN、を読む。確定払戻・walk-forward・6艇クリーン。

例: python -u -m src.test_trifecta_cover
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .predict import harville_trifecta
from .validate import fit_predict_leakfree, SELECTION_YEARS

GRID = [1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 24, 30, 40, 50, 60]


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df):
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if pd.isna(top["trifecta_combo"]):
                continue
            ranked = [c for c, _ in sorted(harville_trifecta(
                {int(l): float(v) for l, v in zip(g["lane"], g["p"])}).items(),
                key=lambda x: -x[1])]
            combo = str(top["trifecta_combo"])
            hitpos = ranked.index(combo) + 1 if combo in ranked else 999  # 何点目で当たるか
            rows.append({
                "honmei": int(top["lane"]), "hitpos": hitpos,
                "yen": float(top["trifecta_yen"]),
            })
    return pd.DataFrame(rows)


def table(d, label):
    n = len(d)
    print(f"\n[{label}] N={n:,}")
    print("  点数 :  的中率   回収率")
    crossed = False
    for k in GRID:
        hit = d[d["hitpos"] <= k]
        hr = len(hit) / n * 100
        roi = hit["yen"].sum() / (n * k * 100) * 100
        star = ""
        if hr >= 50 and not crossed:
            star = "  ← ここで的中率50%超（平均で当たる）"
            crossed = True
        print(f"  {k:3d}点: {hr:5.1f}%   {roi:4.0f}%{star}")


def main():
    df = load()
    d = collect(df)
    print(f"6艇そろい 総数 {len(d):,}（3連単・確定払戻・walk-forward）")
    print("※top-N点=モデルHarville確率の上位N組を買う。的中率=実3連単that上位N点に入る率。")
    table(d, "全体")
    table(d[d["honmei"] == 1], "堅い(本命=1号)")
    table(d[d["honmei"] != 1], "妙味(本命≠1号)")
    print("\n読み: 『平均で当たる』=的中率が50%を超える点数。回収率が100超えなければ、"
          "当たっても控除で負ける(市場効率)。全体の平均的中を見よ。")


if __name__ == "__main__":
    main()
