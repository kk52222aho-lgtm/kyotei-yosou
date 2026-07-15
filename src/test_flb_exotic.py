"""FLB(本命-大穴バイアス)がexoticで100%超えるか＝本命同士のチャラい買い目を狙う。

単勝でFLB確認(大穴27%↔本命94%、本物の歪み)だが単勝は94.7%止まり。FLBはexoticでこそ
濃く出る(本命脚を過小評価→本命-本命の組が過小人気)。∴本命の強さ(model p0)別に、
モデル1位の2連単・3連単チャラ目(=本命脚の組)を1点買いしてROIを測る。控除超え(>100%)が
出れば本物の妙味。特に鉄板(p0高)帯を深掘り。確定払戻・walk-forward・6艇クリーン。

例: python -u -m src.test_flb_exotic
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .predict import harville_trifecta
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.exacta_combo, p.exacta_yen, p.trifecta_combo, p.trifecta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.exacta_yen IS NOT NULL
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
            if len(g) != 6 or pd.isna(g.iloc[0]["exacta_combo"]):
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            p0 = p[int(top["lane"])]
            # モデル1位の2連単(Harville最大)
            h = int(top["lane"]); dh = 1 - p[h]
            ex = max(((f"{h}-{j}", p[h] * p[j] / dh if dh > 0 else 0) for j in p if j != h),
                     key=lambda x: x[1])[0]
            # モデル1位の3連単
            tf = max(harville_trifecta(p).items(), key=lambda x: x[1])[0]
            ex_hit = ex == str(top["exacta_combo"])
            tf_hit = (not pd.isna(top["trifecta_combo"])) and tf == str(top["trifecta_combo"])
            rows.append({
                "p0": p0,
                "ex_ret": float(top["exacta_yen"]) if ex_hit else 0.0, "ex_hit": int(ex_hit),
                "tf_ret": float(top["trifecta_yen"]) if (tf_hit and not pd.isna(top["trifecta_yen"])) else 0.0,
                "tf_ok": not pd.isna(top["trifecta_yen"]), "tf_hit": int(tf_hit),
            })
    return pd.DataFrame(rows)


def line(sub, col, hit, label, ok=None):
    d = sub if ok is None else sub[sub[ok]]
    n = len(d)
    if n < 100:
        print(f"  {label:16s}: n{n} 少")
        return
    roi = d[col].sum() / (n * 100) * 100
    hr = d[hit].mean() * 100
    flag = "  ★控除超え！" if roi >= 100 else ("  ←惜" if roi >= 95 else "")
    print(f"  {label:16s}: 的中{hr:4.1f}%  ROI{roi:5.1f}%  (n{n}){flag}")


def main():
    df = load()
    d = collect(df)
    print(f"6艇そろい 総数 {len(d):,}（モデル1位のチャラ目1点・確定払戻）\n")
    print("[本命の強さ(model p0)別 モデル1位 2連単チャラ目1点]")
    for lo, hi in [(0, .4), (.4, .55), (.55, .7), (.7, 1.01)]:
        line(d[(d["p0"] >= lo) & (d["p0"] < hi)], "ex_ret", "ex_hit", f"p0 {lo:.2f}-{hi:.2f}")
    print("\n[本命の強さ別 モデル1位 3連単チャラ目1点]")
    for lo, hi in [(0, .4), (.4, .55), (.55, .7), (.7, 1.01)]:
        line(d[(d["p0"] >= lo) & (d["p0"] < hi)], "tf_ret", "tf_hit", f"p0 {lo:.2f}-{hi:.2f}", ok="tf_ok")
    print("\n[鉄板帯 深掘り(p0≥閾) 2連単/3連単 1点]")
    for lo in [.6, .65, .7, .75, .8]:
        sub = d[d["p0"] >= lo]
        line(sub, "ex_ret", "ex_hit", f"2連単 p0≥{lo:.2f}")
        line(sub, "tf_ret", "tf_hit", f"3連単 p0≥{lo:.2f}", ok="tf_ok")
    print("\n読み: どこか>100%(★)なら本命バイアスがexoticで控除超え=本物の妙味。")
    print("      全部<100%なら歪みは実在するが控除を超えるほどではない=やはり効率的。")


if __name__ == "__main__":
    main()
