"""「今節モーター」で1号飛んだ後の受益者を当てられるか＝荒れ予想の後半ストーリー再挑戦。

累積motor_2rate(節途中の調子を隠す)でなく、"今節そのモーターが来てるか"(同節の過去レース勝率)を作り、
荒れ予想の本命=1号群(naive/anti-in-prob とも負け=89-90%)で、
『非イン艇のうち今節モーター最強』を単勝で買えば>100%になるかを検証。確定払戻・1点100円。

例: python -u -m src.test_motor_meet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .wild import score_race
from .validate import fit_predict_leakfree, SELECTION_YEARS

TOP_PER_DAY = 12
MIN_MEET = 2   # 今節モーター率を使う最小レース数


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    df["yr"] = df["date"].str[:4]
    return df


def add_motor_meet(df):
    """今節モーター勝率(mm_rate)を各行に付与。リークなし(そのレースより前の同節結果のみ)。"""
    mb = pd.read_csv("data/motor_boat.csv", dtype={"date": str, "jcd": str})
    mb["jcd"] = mb["jcd"].str.zfill(2)
    df = df.merge(mb[["date", "jcd", "rno", "lane", "motor_no"]],
                  on=["date", "jcd", "rno", "lane"], how="left")
    dts = df[["jcd", "date"]].drop_duplicates().copy()
    dts["d"] = pd.to_datetime(dts["date"], format="%Y%m%d")
    dts = dts.sort_values(["jcd", "d"])
    dts["gap"] = dts.groupby("jcd")["d"].diff().dt.days.fillna(99)
    dts["meet_id"] = dts.groupby("jcd")["gap"].transform(lambda s: (s > 1).cumsum())
    df = df.merge(dts[["jcd", "date", "meet_id"]], on=["jcd", "date"], how="left")
    df = df.sort_values(["jcd", "motor_no", "meet_id", "date", "rno"]).reset_index(drop=True)
    g = df.groupby(["jcd", "motor_no", "meet_id"], sort=False)
    wins_before = (g["win"].cumsum() - df["win"]).values
    n_before = g.cumcount().values
    df["mm_n"] = n_before
    df["mm_rate"] = np.where(n_before >= MIN_MEET, wins_before / np.clip(n_before, 1, None), np.nan)
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
    df = add_motor_meet(df)
    print(f"今節モーター付与: {df['mm_rate'].notna().mean()*100:.0f}% の艇が今節{MIN_MEET}走以上\n")

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
            mm = non1.dropna(subset=["mm_rate"])
            best_motor = int(mm.sort_values("mm_rate", ascending=False).iloc[0]["lane"]) if not mm.empty else None
            byday.setdefault(date, []).append({
                "score": sc, "honmei": honmei, "winner": winner, "ty": ty,
                "best_prob": best_prob, "best_motor": best_motor,
            })

    flagged_in1 = []
    for d, rs in byday.items():
        flagged_in1 += [r for r in sorted(rs, key=lambda x: -x["score"])[:TOP_PER_DAY]
                        if r["honmei"] == 1]

    n = len(flagged_in1)
    print(f"荒れ予想・本命=1号の群 {n:,}レース（naive本命単勝89% / anti-in-prob 90% で負けてた所）\n")
    prob = [(r["winner"] == r["best_prob"], r["ty"]) for r in flagged_in1]
    mot = [(r["winner"] == r["best_motor"], r["ty"]) for r in flagged_in1 if r["best_motor"]]
    print(f"  非イン・prob最強 単勝      : {roi(prob)}")
    print(f"  非イン・今節モーター最強 単勝 : {roi(mot)}  <- お前の仮説")
    print("\n※今節モーターthaが>100%なら『1号飛んだ後は今節あってるモーターの艇が来る』=新エッジ。")
    print("  ≈prob(90%)なら今節モーターも織り込み済/効かず。確定払戻・ライブ未証明。")


if __name__ == "__main__":
    main()
