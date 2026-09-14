"""全券種 × 全目 × 全断面 で、回収率100%超のセルを洗い出す。

理屈は一切使わん。予測も使わん。**過去の払戻を数えるだけ。**

券種: 単勝6 / 2連複15 / 2連単30 / 3連複20 / 3連単120 = 191目
断面: 全体 / 場24 / R番号12 / 風速帯 / 波高帯 / 月12 …と、場×R(288)まで

出すもん(全部同じ表に入れる。選り好みせん):
  1. **回収率100%超のセル**を全部
  2. **偶然やったら何本出るか**(場ラベルをシャッフルした帰無での本数)
  3. **前半(2022-2024)で100%超えたセルが、後半(2025-2026)でも超えたか**

3が一番大事や。理屈が要らんのはそのとおりで、**要るのは「翌年も来たか」だけ**やから。

usage:
  python -u -m src.scan_over100
出力: data/over100.csv(全セル) / data/over100_hits.csv(100%超のみ)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import storage
from .venues import name as venue_name

RNG = np.random.default_rng(20260902)
SPLIT = "20250101"          # 前半/後半の境目
MIN_BETS = 30               # これ未満は表に出すが別枠(n が小さすぎる)

BETS = [
    ("単勝", "tansho_lane", "tansho_yen", 1),
    ("2連複", "quinella_combo", "quinella_yen", 2),
    ("2連単", "exacta_combo", "exacta_yen", 2),
    ("3連複", "trio_combo", "trio_yen", 3),
    ("3連単", "trifecta_combo", "trifecta_yen", 3),
]


def load():
    conn = storage.connect()
    p = pd.read_sql_query("SELECT * FROM payouts", conn)
    e = pd.read_sql_query("""
        SELECT date, jcd, rno, COUNT(*) n,
               MAX(wind_speed) wind, MAX(wave_height) wave
        FROM entries GROUP BY date, jcd, rno""", conn)
    conn.close()
    for d in (p, e):
        d["jcd"] = d["jcd"].astype(str).str.zfill(2)
    df = p.merge(e, on=["date", "jcd", "rno"], how="inner")
    df = df[df["n"] == 6].drop(columns="n").reset_index(drop=True)
    df["tansho_lane"] = df["tansho_lane"].astype("Int64").astype(str)
    df["yr"] = df["date"].str[:4]
    df["late"] = (df["date"] >= SPLIT).astype(int)
    df["場"] = df["jcd"].map(venue_name)
    df["R"] = "R" + df["rno"].astype(str).str.zfill(2)
    df["月"] = df["date"].str[4:6] + "月"
    df["風"] = pd.cut(pd.to_numeric(df["wind"], errors="coerce"),
                      [-0.1, 1, 3, 5, 99],
                      labels=["風0-1m", "風2-3m", "風4-5m", "風6m+"]).astype(str)
    df["波"] = pd.cut(pd.to_numeric(df["wave"], errors="coerce"),
                      [-0.1, 1, 3, 5, 999],
                      labels=["波0-1cm", "波2-3cm", "波4-5cm", "波6cm+"]).astype(str)
    df["場R"] = df["場"] + df["R"]
    return df


def cells(df, segcol, label):
    """1つの断面について、全券種×全目のセルを返す。"""
    out = []
    nrace = df.groupby(segcol).size()
    nlate = df[df.late == 1].groupby(segcol).size()
    nearly = df[df.late == 0].groupby(segcol).size()
    for bname, ccol, ycol, _ in BETS:
        d = df[[segcol, ccol, ycol, "late"]].dropna(subset=[ccol, ycol])
        g = d.groupby([segcol, ccol])[ycol].agg(["sum", "count"])
        gl = d[d.late == 1].groupby([segcol, ccol])[ycol].agg(["sum", "count"])
        ge = d[d.late == 0].groupby([segcol, ccol])[ycol].agg(["sum", "count"])
        for (seg, combo), r in g.iterrows():
            n = nrace.get(seg, 0)
            if n < 200:
                continue
            ne, nl = nearly.get(seg, 0), nlate.get(seg, 0)
            se = ge["sum"].get((seg, combo), 0.0)
            sl = gl["sum"].get((seg, combo), 0.0)
            out.append({
                "断面": label, "セグ": str(seg), "券種": bname, "目": str(combo),
                "レース": int(n), "的中": int(r["count"]),
                "出現率": r["count"] / n,
                "回収率": r["sum"] / (n * 100),
                "前半回収": se / (ne * 100) if ne else np.nan,
                "後半回収": sl / (nl * 100) if nl else np.nan,
                "前半的中": int(ge["count"].get((seg, combo), 0)),
                "後半的中": int(gl["count"].get((seg, combo), 0)),
            })
    return out


def main():
    df = load()
    print(f"盤: {len(df):,}レース / {df['date'].nunique():,}日 "
          f"({df['date'].min()}-{df['date'].max()})  6艇そろいのみ")
    print(f"前半 {(df.late == 0).sum():,} / 後半 {(df.late == 1).sum():,}"
          f"(境目 {SPLIT})\n")

    df["全体"] = "全場"
    rows = []
    for col, label in [("全体", "全体"), ("場", "場"), ("R", "R番号"),
                       ("月", "月"), ("風", "風速帯"), ("波", "波高帯"),
                       ("場R", "場×R")]:
        c = cells(df, col, label)
        rows += c
        print(f"  {label:8s} セル {len(c):>7,}")
    C = pd.DataFrame(rows)
    C.to_csv("data/over100.csv", index=False, encoding="utf-8-sig")
    print(f"\n総セル数 {len(C):,}")

    hit = C[C["回収率"] > 1.0].sort_values("回収率", ascending=False)
    hit.to_csv("data/over100_hits.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 72}")
    print(f"■ 回収率100%超のセル: {len(hit):,} / {len(C):,}  ({len(hit) / len(C):.2%})")
    print(f"{'=' * 72}")
    big = hit[hit["的中"] >= MIN_BETS]
    print(f"  うち的中{MIN_BETS}本以上: {len(big):,}")
    print(f"  うち的中 5本未満  : {(hit['的中'] < 5).sum():,}  ← 1本2本の当たりで100%超えとるやつ")

    print(f"\n--- 回収率 上位25(的中{MIN_BETS}本以上) ---")
    print(f"  {'断面':<6}{'セグ':<12}{'券種':<6}{'目':<8}{'レース':>7}"
          f"{'的中':>6}{'回収率':>8}{'前半':>8}{'後半':>8}")
    for _, r in big.head(25).iterrows():
        print(f"  {r['断面']:<6}{r['セグ']:<12}{r['券種']:<6}{r['目']:<8}"
              f"{r['レース']:>7,}{r['的中']:>6}{r['回収率']:>8.1%}"
              f"{r['前半回収']:>8.1%}{r['後半回収']:>8.1%}")

    # ---- 偶然やったら何本出るか(場ラベルをシャッフル) ----
    print(f"\n--- 偶然の本数(場ラベルをシャッフルして同じ数え方) ---")
    counts = []
    base = df[["jcd", "場"]].drop_duplicates()
    for i in range(20):
        t = df.copy()
        t["場"] = RNG.permutation(df["場"].to_numpy())
        c = pd.DataFrame(cells(t, "場", "場"))
        counts.append(int((c["回収率"] > 1.0).sum()))
    real = int((C[C["断面"] == "場"]["回収率"] > 1.0).sum())
    print(f"  実測(場断面) {real} 本 vs 帰無 中央値 {int(np.median(counts))} 本 "
          f"(範囲 {min(counts)}-{max(counts)})")

    # ---- 前半で100%超えたやつは、後半も超えたか ----
    print(f"\n{'=' * 72}")
    print("■ 前半(2022-2024)で100%超えたセルは、後半(2025-2026)でも超えたか")
    print(f"{'=' * 72}")
    for nmin in (0, 10, 30, 100):
        e = C[(C["前半回収"] > 1.0) & (C["前半的中"] >= nmin)].dropna(subset=["後半回収"])
        if not len(e):
            continue
        keep = (e["後半回収"] > 1.0).mean()
        print(f"  前半的中{nmin:>4}本以上: 前半100%超 {len(e):>6,}本 → "
              f"後半も100%超 {int((e['後半回収'] > 1.0).sum()):>5,}本 ({keep:.1%})  "
              f"後半の平均回収 {e['後半回収'].mean():.1%}")
    print("\n  ※ 全セルの後半平均回収率(参考):",
          f"{C['後半回収'].mean():.1%}")
    print("\n保存: data/over100.csv / data/over100_hits.csv")


if __name__ == "__main__":
    main()
