"""残り全券種(3連複/3連単/高確信)もサバイバーシップ除去後(6艇そろい)で測り直す＝地図完成。

survivorshipで単勝/2連単3点はクリーン(6艇)で95%/82%=死と判明。残りの汚染ROI主張
(3連複4点/3連単3点/高確信p0>=45%スライス)も同じ6艇そろいだけで測り、
汚染(全体) vs クリーン(6艇) を並べて何that本物か何that飛び艇水増しか確定させる。
確定払戻・walk-forward・harville。

例: python -u -m src.test_clean_all
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen, p.exacta_combo, p.exacta_yen,
               p.trio_combo, p.trio_yen, p.trifecta_combo, p.trifecta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.exacta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def harville_trifecta(p):
    out = []
    for i in p:
        di = 1 - p[i]
        if di <= 0:
            continue
        for j in p:
            if j == i:
                continue
            dij = 1 - p[i] - p[j]
            if dij <= 0:
                continue
            for k in p:
                if k in (i, j):
                    continue
                out.append((f"{i}-{j}-{k}", p[i] * p[j] / di * p[k] / dij))
    return sorted(out, key=lambda x: -x[1])


def trio_rank(p):
    seen, out = set(), []
    for c, _ in harville_trifecta(p):
        key = "=".join(sorted(c.split("-"), key=int))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


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
            honmei = int(top["lane"])
            if honmei == 1 or pd.isna(top["exacta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            clean = bool((g["lane"] == 1).any()) and len(g) == 6      # ライブ再現可=6艇そろい

            second = [int(l) for l in g["lane"] if int(l) != honmei][:3]
            ex_hit = str(top["exacta_combo"]) in {f"{honmei}-{x}" for x in second}
            trio_buys = trio_rank(p)[:4]
            trio_hit = str(top["trio_combo"]) in trio_buys if not pd.isna(top["trio_combo"]) else False
            tf_buys = [c for c, _ in harville_trifecta(p)][:3]
            tf_hit = str(top["trifecta_combo"]) in tf_buys if not pd.isna(top["trifecta_combo"]) else False
            tan_hit = (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == honmei

            rows.append({
                "yr": y, "clean": clean, "p0": p[honmei],
                "tan_ret": float(top["tansho_yen"]) if tan_hit else 0.0,
                "ex_ret": float(top["exacta_yen"]) if ex_hit else 0.0,
                "trio_ret": float(top["trio_yen"]) if (trio_hit and not pd.isna(top["trio_yen"])) else 0.0,
                "trio_ok": not pd.isna(top["trio_yen"]),
                "tf_ret": float(top["trifecta_yen"]) if (tf_hit and not pd.isna(top["trifecta_yen"])) else 0.0,
                "tf_ok": not pd.isna(top["trifecta_yen"]),
            })
    return pd.DataFrame(rows)


def roi(sub, col, pts):
    n = len(sub)
    return (sub[col].sum() / (n * pts * 100) * 100) if n else 0


def block(d, label):
    n = len(d)
    if not n:
        print(f"  {label:24s}: 0件")
        return
    dt = d[d["trio_ok"]]
    df_ = d[d["tf_ok"]]
    print(f"  {label:22s} N={n:5d}: 単勝{roi(d,'tan_ret',1):4.0f}%  2連単3点{roi(d,'ex_ret',3):4.0f}%  "
          f"3連複4点{roi(dt,'trio_ret',4):4.0f}%  3連単3点{roi(df_,'tf_ret',3):4.0f}%")


def main():
    df = load()
    d = collect(df)
    print(f"本命≠1号(妙味) 総数 {len(d):,}\n")
    print("[全券種: 汚染(全体) vs クリーン(6艇そろい=ライブ再現可)]")
    block(d, "汚染(全体)")
    block(d[d["clean"]], "クリーン(6艇)")
    block(d[~d["clean"]], "飛び艇あり(非クリーン)")

    print("\n[高確信 p0>=45%(最強妙味の確率条件)を クリーン(6艇)だけで]")
    hc = d[d["clean"] & (d["p0"] >= 0.45)]
    block(hc, "クリーン×p0>=45%")
    print("  (最強妙味592%=本命≠1号×p0>=45%×2連単EV>3.5。EV条件は高配当=飛び艇selectionでさらに汚染。")
    print("   ここのクリーンp0>=45%that100%割れなら最強妙味も水増しで確定)")

    print("\n[クリーン(6艇)・年別]")
    for y in sorted(d["yr"].unique()):
        block(d[d["clean"] & (d["yr"] == y)], f"{y}")

    print("\n判定: クリーン(6艇)で全券種<100%なら、券種を問わずコアは飛び艇survivorshipの水増しで確定。")
    print("※確定払戻・walk-forward・harville。3連系はharville近似の買い目選択。")


if __name__ == "__main__":
    main()
