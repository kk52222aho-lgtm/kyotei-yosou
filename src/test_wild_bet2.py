"""荒れ予想レースの「1号飛んでからのストーリー」＝受益者(一番強い非イン艇)を買えたか。

前作(test_wild_bet)は本命単勝で見て甘かった: 本命=1号の群は"飛ぶ側の1号"に賭けてて負け。
ここは「1号が飛ぶ前提で、受益者＝最強の非イン艇(lane2-6でwin_prob最大)を単勝で買う」を検証。
naive(本命単勝) vs anti-in(最強非イン単勝) を、本命=1号/≠1号で分解。確定払戻・1点100円。

例: python -u -m src.test_wild_bet2
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .wild import score_race
from .validate import fit_predict_leakfree, SELECTION_YEARS

TOP_PER_DAY = 12


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
            ty = float(top["tansho_yen"])
            non1 = [l for l in p if l != 1]
            best_non1 = max(non1, key=lambda l: p[l]) if non1 else honmei
            rows.append({
                "date": date, "score": w["score"], "honmei": honmei, "winner": winner,
                "tan_honmei": ty if winner == honmei else 0.0,
                "best_non1": best_non1,
                "tan_bn1": ty if winner == best_non1 else 0.0,
            })
    return rows


def roi(grp, retkey, pickkey):
    m = len(grp)
    if not m:
        return "0件"
    ret = sum(r[retkey] for r in grp)
    hit = sum(1 for r in grp if r["winner"] == r[pickkey])
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
    in1 = [r for r in flagged if r["honmei"] == 1]
    ne1 = [r for r in flagged if r["honmei"] != 1]
    print(f"荒れ予想(各日top{TOP_PER_DAY}) {n:,}レース｜1号飛び {broke*100:.0f}%\n")
    print("  [naive = 本命の単勝（前作・甘い）]")
    print(f"     全フラグ  : {roi(flagged, 'tan_honmei', 'honmei')}")
    print(f"     本命=1号  : {roi(in1, 'tan_honmei', 'honmei')}")
    print(f"     本命!=1号 : {roi(ne1, 'tan_honmei', 'honmei')}")
    print("\n  [anti-in = 1号飛ぶ前提で受益者=最強の非イン艇の単勝]")
    print(f"     全フラグ  : {roi(flagged, 'tan_bn1', 'best_non1')}")
    print(f"     本命=1号  : {roi(in1, 'tan_bn1', 'best_non1')}  <- naiveで負けてた1万件が買えるか")
    print(f"     本命!=1号 : {roi(ne1, 'tan_bn1', 'best_non1')}")
    print("\n※本命=1号群で anti-in that>100%なら『1号飛んでからのストーリー』は本物=受益者を買えばよかった。")
    print("  確定払戻・ライブ未証明。薄プール自己インパクトは別途。")


if __name__ == "__main__":
    main()
