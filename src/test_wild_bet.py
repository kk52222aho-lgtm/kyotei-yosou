"""過去3年で「荒れそう予想が本当に荒れた率」＋「買えたか（本命単勝の勝敗）」。

ライブ検証ログ(wild_log)は明日から。ここは過去分で今すぐ答える。
各日の荒れ度top12(=ページのフラグと同じ)をクリーン年で拾い:
  ①ほんとに荒れた率(1号飛び=winner≠1) ②万舟率(3連単≥1万)
  ③買えたか＝本命単勝ROI。全フラグ/本命≠1号(実際に張る妙味)/本命=1号(イン賭け)で分解。
確定払戻・1点100円。

例: python -u -m src.test_wild_bet
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .wild import score_race
from .validate import fit_predict_leakfree, SELECTION_YEARS

TOP_PER_DAY = 12
MANSHU = 10000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_yen, p.tansho_lane, p.tansho_yen
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
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if pd.isna(top["tansho_lane"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            srows = [{"lane": int(r.lane), "win_pct": p[int(r.lane)] * 100,
                      "racer_class": r.racer_class} for r in g.itertuples()]
            w = score_race(srows)
            honmei = int(top["lane"])
            winner = int(top["tansho_lane"])
            rows.append({
                "date": date, "score": w["score"], "honmei": honmei, "winner": winner,
                "tan_ret": float(top["tansho_yen"]) if winner == honmei else 0.0,
                "tf_yen": float(top["trifecta_yen"]),
            })
    return rows


def tan(grp):
    m = len(grp)
    if not m:
        return "0件"
    ret = sum(r["tan_ret"] for r in grp)
    hit = sum(1 for r in grp if r["winner"] == r["honmei"])
    return f"{m}件 的中{hit/m*100:.0f}% 回収{ret/(m*100)*100:.0f}% 収支{ret-m*100:+,.0f}円"


def main():
    df = load()
    rows = collect(df)
    byday = {}
    for r in rows:
        byday.setdefault(r["date"], []).append(r)
    flagged = []
    for d, rs in byday.items():
        flagged += sorted(rs, key=lambda x: -x["score"])[:TOP_PER_DAY]

    n = len(flagged)
    broke = sum(1 for r in flagged if r["winner"] != 1) / n
    man = sum(1 for r in flagged if r["tf_yen"] >= MANSHU) / n
    base_broke = sum(1 for r in rows if r["winner"] != 1) / len(rows)
    print(f"クリーン年・各日 荒れ度top{TOP_PER_DAY} ＝ 荒れそう予想 {n:,}レース\n")
    tag = ("当ててる +" + str(round((broke - base_broke) * 100)) + "pt") if broke > base_broke else "外し"
    print(f"  1 ほんとに荒れた率(1号飛び)：{broke*100:.0f}%   （全レース基準 {base_broke*100:.0f}% → {tag}）")
    print(f"  2 万舟率(3連単>=1万)：{man*100:.0f}%   （条件不問の基準≈24%とほぼ同じ＝万舟は事前に読めん）")
    print(f"\n  3 買えたか＝本命の単勝で買った場合（1点100円）")
    print(f"     全フラグ                 : {tan(flagged)}")
    print(f"     本命!=1号(実際に張る妙味) : {tan([r for r in flagged if r['honmei'] != 1])}")
    print(f"     本命=1号(イン賭け=分悪)   : {tan([r for r in flagged if r['honmei'] == 1])}")
    print("\n※確定払戻・ライブ未証明。荒れ予想は1号飛びは当てるが万舟は読めん。")
    print("  買うなら本命!=1号サブセットだけ＝それが妙味＝単勝116%の中身。全フラグを本命単勝で買うのは分が悪い。")


if __name__ == "__main__":
    main()
