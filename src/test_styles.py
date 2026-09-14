"""買い方の"アクセル段階"別トータル比較（堅実4点〜万舟全張り120点）。

ユーザー要望: 「20倍20点、万舟狙いなら100点、みたいな買い方のトータルthaが知りたい」。
妙味(本命≠1号)レースで、点数を段階的に増やした各スタイルの
  総賭金/総払戻/回収率/レース単位的中率/万舟(3連単>=1万)捕捉数/最大払戻 を並べる。
※確定払戻ベース=ライブ未証明。点を増やすほど回収率は下がる(控除)一方、的中率と万舟捕捉は上がる
  =「当てる回数↑・効率↓」のトレードを数字で見る用。1点=100円。

例: python -u -m src.test_styles
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import storage
from .predict import harville_trifecta
from .strategy_search import trifecta_probs
from .validate import fit_predict_leakfree, SELECTION_YEARS

MANSHU = 10000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.trio_combo, p.trio_yen,
               p.exacta_combo, p.exacta_yen, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def trio_rank(p):
    agg = defaultdict(float)
    for c, pr in trifecta_probs(p):
        agg["-".join(sorted(c.split("-"), key=int))] += pr
    return [c for c, _ in sorted(agg.items(), key=lambda x: -x[1])]


def collect(df):
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for _, g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["trifecta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            ex = []
            for i in p:
                di = 1 - p[i]
                if di > 0:
                    for j in p:
                        if j != i:
                            ex.append((f"{i}-{j}", p[i] * p[j] / di))
            ex.sort(key=lambda x: -x[1])
            rows.append({
                "won_t": (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == int(top["lane"]),
                "tan_yen": float(top["tansho_yen"]) if not pd.isna(top["tansho_yen"]) else 0.0,
                "ex_rank": [c for c, _ in ex], "ex_combo": str(top["exacta_combo"]),
                "ex_yen": float(top["exacta_yen"]) if not pd.isna(top["exacta_yen"]) else 0.0,
                "tr_rank": trio_rank(p), "tr_combo": str(top["trio_combo"]), "tr_yen": float(top["trio_yen"]),
                "tf_rank": [c for c, _ in sorted(harville_trifecta(p).items(), key=lambda x: -x[1])],
                "tf_combo": str(top["trifecta_combo"]), "tf_yen": float(top["trifecta_yen"]),
            })
    return rows


# スタイル = 各賭式の点数 (tansho,exacta,trio,trifecta)。tanshoは1(本命)固定。
STYLES = [
    ("堅実 4点(単1+2連3)",       1, 3, 0, 0),
    ("標準 11点(+3複4+3単3)",     1, 3, 4, 3),
    ("3連複ワイド 20点(3複全)",    0, 0, 20, 0),
    ("中 3連単20点狙い",          0, 0, 0, 20),
    ("積極 3連単60点",           0, 0, 0, 60),
    ("万舟全張り 3連単120点",      1, 0, 0, 120),
]


def evaluate(rows, tan, exn, trn, tfn):
    n = len(rows)
    stake = ret = hits = manshu = 0
    mx = 0
    for r in rows:
        rr = 0.0
        if tan:
            stake += tan * 100
            if r["won_t"]:
                rr += r["tan_yen"]
        if exn:
            stake += exn * 100
            if r["ex_combo"] in r["ex_rank"][:exn]:
                rr += r["ex_yen"]
        if trn:
            stake += trn * 100
            if r["tr_combo"] in r["tr_rank"][:trn]:
                rr += r["tr_yen"]
        if tfn:
            stake += tfn * 100
            if r["tf_combo"] in r["tf_rank"][:tfn]:
                rr += r["tf_yen"]
                if r["tf_yen"] >= MANSHU:
                    manshu += 1
        ret += rr
        if rr > 0:
            hits += 1
        mx = max(mx, rr)
    pts = tan + exn + trn + tfn
    return {"pts": pts, "stake": stake, "ret": ret, "roi": ret / stake if stake else 0,
            "pl": ret - stake, "hit": hits / n, "manshu": manshu, "mx": mx}


def main():
    df = load()
    rows = collect(df)
    n = len(rows)
    print(f"妙味(本命≠1号)・クリーン年 {n:,}レース｜1点100円｜万舟=3連単>={MANSHU:,}\n")
    print(f"  {'スタイル':<24}{'点/R':>5}{'総賭金':>12}{'回収率':>7}{'総収支':>13}{'的中率':>7}{'万舟':>5}{'最大':>9}")
    for name, ta, ex, trn, tf in STYLES:
        s = evaluate(rows, ta, ex, trn, tf)
        print(f"  {name:<24}{s['pts']:>5}{s['stake']:>11,.0f}{s['roi']*100:>6.0f}%"
              f"{s['pl']:>+12,.0f}{s['hit']*100:>6.0f}%{s['manshu']:>5}{s['mx']:>8,.0f}")
    print("\n※読み: 点を増やすほど『的中率↑・万舟捕捉↑』だが『回収率↓・総収支悪化』(控除を余計に払う)。")
    print("  全張り系(120点)は的中率ほぼ100%でも回収率が控除線に張り付く＝当ててるのに負ける典型。")
    print("  効率(回収率)は点を絞る方が上。広く買うのは『当てる快感』を金で買う行為。確定払戻=ライブ未証明。")


if __name__ == "__main__":
    main()
