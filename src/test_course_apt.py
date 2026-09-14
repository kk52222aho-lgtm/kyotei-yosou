"""選手のコース適性(個性)で1号飛んだ後の受益者を当てられるか。

全国勝率は全コース平均＝「2コース差しの名手」みたいな個性が消える。
reg×lane×finish から『この選手のこの枠からの勝率(コース適性)』を作り(leak-free)、
本命=1号の荒れ予想(単勝89-90%で負けてた群)で、非イン艇のうち"今いる枠の適性が最強"を
単勝で買えば>100%になるか。prob/motor選抜(90/81%)を超えるか。確定払戻・1点100円。

例: python -u -m src.test_course_apt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .wild import score_race
from .validate import fit_predict_leakfree, SELECTION_YEARS

TOP_PER_DAY = 12
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


def add_course_apt(df):
    df["_w"] = (df["finish"].fillna(99) == 1).astype(float)
    df = df.sort_values(["reg", "lane", "date", "rno"]).reset_index(drop=True)
    g = df.groupby(["reg", "lane"], sort=False)
    wins_before = (g["_w"].cumsum() - df["_w"]).values
    n_before = g.cumcount().values
    df["apt_n"] = n_before
    df["apt"] = np.where(n_before >= MIN_APT, wins_before / np.clip(n_before, 1, None), np.nan)
    return df


def roi(picks):
    m = len(picks)
    if not m:
        return "0件"
    ret = sum(y for w, y in picks if w)
    hit = sum(1 for w, y in picks if w)
    return f"{m}件 的中{hit/m*100:.0f}% 回収{ret/(m*100)*100:.0f}% 収支{ret-m*100:+,.0f}円"


def main():
    df = load()
    df = add_course_apt(df)
    print(f"コース適性付与: {df['apt'].notna().mean()*100:.0f}% の艇が該当枠{MIN_APT}走以上\n")

    byday = {}
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
            sc = score_race(srows)["score"]
            honmei = int(top["lane"])
            winner = int(top["tansho_lane"])
            ty = float(top["tansho_yen"])
            non1 = g[g["lane"] != 1]
            if non1.empty:
                continue
            best_prob = int(non1.iloc[0]["lane"])
            mm = non1.dropna(subset=["apt"])
            best_apt = int(mm.sort_values("apt", ascending=False).iloc[0]["lane"]) if not mm.empty else None
            byday.setdefault(date, []).append({
                "score": sc, "honmei": honmei, "winner": winner, "ty": ty,
                "best_prob": best_prob, "best_apt": best_apt,
            })

    in1 = []
    for d, rs in byday.items():
        in1 += [r for r in sorted(rs, key=lambda x: -x["score"])[:TOP_PER_DAY] if r["honmei"] == 1]

    print(f"荒れ予想・本命=1号 {len(in1):,}レース（prob選抜90% / motor選抜81% で負けてた群）\n")
    prob = [(r["winner"] == r["best_prob"], r["ty"]) for r in in1]
    apt = [(r["winner"] == r["best_apt"], r["ty"]) for r in in1 if r["best_apt"]]
    print(f"  非イン・prob最強 単勝       : {roi(prob)}")
    print(f"  非イン・コース適性最強 単勝 : {roi(apt)}  <- 個性(差し得意度)で受益者を選ぶ")
    print("\n※コース適性が>100%なら『1号飛んだ後はその枠が得意な選手が来る』＝個性で受益者が読める新エッジ。")
    print("  ≈prob(90%)なら個性も織り込み済/効かず。確定払戻・ライブ未証明。")


if __name__ == "__main__":
    main()
