"""1号飛んだ後のシナリオ再詰め: 単勝でなくエキゾチック(1号を外した2連単/3連単/3連複)で買えたか。

単勝は全滅(89-90%)＝市場が単勝を正しく値付け。だが今日の唯一の勝ち筋は2連単EVミスプライス
＝群衆が目の重み付けを間違える。∴1号飛ぶ荒れ予想でも、群衆が弱い1号を未だ買い目に入れてる分、
"1号を完全に外した非イン目"が割安な可能性。本命=1号の荒れ予想(単勝で負けてた群)で、
非イン限定の2連単/3連単/3連複を確定払戻で検証。1点100円・妙味top12/日。

例: python -u -m src.test_after_flies
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
        SELECT e.*, p.exacta_combo, p.exacta_yen, p.trio_combo, p.trio_yen,
               p.trifecta_combo, p.trifecta_yen, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def exacta_non1(p):
    non1 = {l: p[l] for l in p if l != 1}
    out = []
    for i in non1:
        di = 1 - non1[i]
        if di > 0:
            for j in non1:
                if j != i:
                    out.append((f"{i}-{j}", non1[i] * non1[j] / di))
    return [c for c, _ in sorted(out, key=lambda x: -x[1])]


def trifecta_non1(p):
    non1 = {l: p[l] for l in p if l != 1}
    out = []
    ls = list(non1)
    for i in ls:
        di = 1 - non1[i]
        if di <= 0:
            continue
        for j in ls:
            if j == i:
                continue
            dij = 1 - non1[i] - non1[j]
            if dij <= 0:
                continue
            for k in ls:
                if k in (i, j):
                    continue
                out.append((f"{i}-{j}-{k}", non1[i] * non1[j] / di * non1[k] / dij))
    return [c for c, _ in sorted(out, key=lambda x: -x[1])]


def trio_non1(p):
    seen, out = set(), []
    for c in trifecta_non1(p):
        k = "-".join(sorted(c.split("-"), key=int))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def collect(df):
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
            byday.setdefault(date, []).append({
                "score": score_race(srows)["score"], "honmei": int(top["lane"]),
                "ex_rank": exacta_non1(p), "ex_combo": str(top["exacta_combo"]),
                "ex_yen": float(top["exacta_yen"]) if not pd.isna(top["exacta_yen"]) else 0.0,
                "tf_rank": trifecta_non1(p), "tf_combo": str(top["trifecta_combo"]),
                "tf_yen": float(top["trifecta_yen"]),
                "tr_rank": trio_non1(p), "tr_combo": str(top["trio_combo"]),
                "tr_yen": float(top["trio_yen"]),
            })
    flagged = []
    for d, rs in byday.items():
        flagged += sorted(rs, key=lambda x: -x["score"])[:TOP_PER_DAY]
    return flagged


def roi(rows, rankkey, combokey, yenkey, k):
    n = len(rows)
    ret = sum(r[yenkey] for r in rows if r[combokey] in r[rankkey][:k])
    hit = sum(1 for r in rows if r[combokey] in r[rankkey][:k])
    return f"{k}点:的中{hit/n*100:.0f}%/回収{ret/(n*k*100)*100:.0f}%"


def main():
    df = load()
    flagged = collect(df)
    in1 = [r for r in flagged if r["honmei"] == 1]
    print(f"荒れ予想・本命=1号 {len(in1):,}レース（単勝で89%負けてた群）｜1号を完全に外したエキゾチック\n")
    print("  非イン2連単:", " ".join(roi(in1, "ex_rank", "ex_combo", "ex_yen", k) for k in [1, 3, 6]))
    print("  非イン3連複:", " ".join(roi(in1, "tr_rank", "tr_combo", "tr_yen", k) for k in [1, 2, 4]))
    print("  非イン3連単:", " ".join(roi(in1, "tf_rank", "tf_combo", "tf_yen", k) for k in [3, 6, 12]))
    print("\n※どれか>100%なら『1号飛んだ後は非イン限定エキゾチックで拾える』＝新シナリオ。")
    print("  全部<100%なら市場が非イン目も正しく値付け＝金にならんで確定。確定払戻・ライブ未証明。")


if __name__ == "__main__":
    main()
