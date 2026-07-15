"""6ptの壁を埋める: FLBが構造的に濃くなる断面を、行動メカ裏打ち×3年一貫で探す。

全本命ベタ買いは94.7%が天井で平ら=同じ買い方では埋まらん。埋めるなら本命の過小人気が
構造的にデカくなる断面。小標本の蜃気楼を避け、①行動メカに裏打ち②3年全部で100%超え を条件:
  rno別(最終Rは負け組が取り返しで大穴へ→本命さらに過小人気=chase betting・競馬で実証)
  場のイン天国度別(大村等イン濃い場で本命の値付き)
  本命がインか外か
本命=モデルrank1・model_p≥0.55の単勝。確定払戻・walk-forward・6艇クリーン。年別ROIも必ず出す。

例: python -u -m src.test_flb_close
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS

# イン天国度(2016+実測イン勝率のおおまかな3群)
IN_HEAVEN = {24, 22, 21, 18}          # 大村/福岡/芦屋/徳山 等イン濃い
IN_HELL = {2, 3, 10, 11}              # 戸田/江戸川/三国/びわこ 等イン薄い


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
            s = g["p"].sum()
            if s <= 0:
                continue
            p0 = float(top["p"]) / s
            if p0 < 0.55:                     # 本命帯だけ(FLBの利く側)
                continue
            won = int(top["tansho_lane"]) == int(top["lane"])
            rows.append({
                "yr": y, "rno": int(rno), "jcd": int(jcd),
                "honmei_in": int(top["lane"]) == 1, "p0": p0,
                "won": int(won), "ret": float(top["tansho_yen"]) if won else 0.0,
            })
    return pd.DataFrame(rows)


def roi(sub):
    n = len(sub)
    return (sub["ret"].sum() / (n * 100) * 100) if n else 0, n


def show(d, mask, label):
    sub = d[mask]
    r, n = roi(sub)
    if n < 300:
        print(f"  {label:20s}: n{n} 少")
        return
    ys = []
    ok = True
    for y in sorted(d["yr"].unique()):
        ry, ny = roi(sub[sub["yr"] == y])
        ys.append(f"{y}:{ry:.0f}%")
        if ry < 100:
            ok = False
    flag = "  ★3年全部100%超！" if ok else ("  ←全体は超" if r >= 100 else "")
    print(f"  {label:20s}: ROI{r:5.1f}% (n{n})  [{' '.join(ys)}]{flag}")


def main():
    df = load()
    d = collect(df)
    print(f"本命帯(model_p≥0.55) 総数 {len(d):,}  全体ROI {roi(d)[0]:.1f}%\n")

    print("[レース番号別(最終Rのchase betting狙い)]")
    for rn in range(1, 13):
        show(d, d["rno"] == rn, f"{rn}R")

    print("\n[場のイン天国度別]")
    show(d, d["jcd"].isin(IN_HEAVEN), "イン天国場")
    show(d, d["jcd"].isin(IN_HELL), "イン地獄場")
    show(d, ~d["jcd"].isin(IN_HEAVEN | IN_HELL), "中間場")

    print("\n[本命がインか外か]")
    show(d, d["honmei_in"], "本命=イン(1号)")
    show(d, ~d["honmei_in"], "本命=外(2-6号)")

    print("\n[複合: 最終R×イン天国 / 最終R×本命外 等 濃い所]")
    show(d, (d["rno"] >= 11) & d["jcd"].isin(IN_HEAVEN), "11-12R×イン天国")
    show(d, (d["rno"] >= 11) & ~d["honmei_in"], "11-12R×本命外")
    show(d, (d["rno"] >= 10) & (d["p0"] >= 0.7), "10-12R×超本命p0≥.7")

    print("\n判定: ★(3年全部100%超)が出れば構造的な歪み=本物の妙味。全部<100%なら6ptは埋まらん。")


if __name__ == "__main__":
    main()
