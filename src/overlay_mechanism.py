"""黒箱の生き残りに、機構の変数を重ねる。

`scan_over100` が出した「前半も後半も回収100%超」のセルは、説明無しの暗黙知や。
そこに **場×R ごとの構造**(イン勝率・前づけ率・進入の乱れ・風・波・メンバー構成)を重ねて、
生き残りが構造の上に乗っとるかを見る。

問いは1つ:
  **前半で100%超えたセルのうち、どれが後半も超えたか — それを構造で当てられるか。**

当てられる → 黒箱に機構がある。理屈が黒箱を選別できる = 前向きの母集団を絞れる
当てられん → 生き残りは裾の揺れ。理屈を重ねても救えん

理屈で「なんで効くか」を説明させるんやない。**理屈が予測に使えるかだけ**を測る。
説明は要らんが、予測に使えるなら道具になる([[project_tacit_edge]])。

構造の変数(全部レース前に見えるもん + 進入は事後やが場の性質として使う):
  in_win     1号艇の勝率(その場R)
  maegumi    前づけ率(進入 != 艇番)          ← results_st。今日まで盤に無かった量
  wakunari   全6艇が枠なり進入やったレースの率
  st_sd      レース内STのばらつき(平均)
  wind/wave  平均風速・波高
  a1         A1級の比率
  pow_gap    1号艇の全国勝率 − 他5艇の平均

usage: python -u -m src.overlay_mechanism
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .venues import name as venue_name

RNG = np.random.default_rng(20260902)
MIN_HIT = 10          # 前半の的中本数の下限(これ未満は数えん)


def structure():
    """場×R ごとの構造変数。"""
    conn = storage.connect()
    e = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.finish, e.racer_class, e.nat_win,
               e.wind_speed, e.wave_height, s.course, s.st
        FROM entries e LEFT JOIN results_st s
          ON e.date=s.date AND e.jcd=s.jcd AND e.rno=s.rno AND e.lane=s.lane
    """, conn)
    conn.close()
    e["jcd"] = e["jcd"].astype(str).str.zfill(2)
    sz = e.groupby(["date", "jcd", "rno"])["lane"].transform("size")
    e = e[sz == 6].copy()
    e["場R"] = e["jcd"].map(venue_name) + "R" + e["rno"].astype(str).str.zfill(2)

    e["win1"] = ((e["lane"] == 1) & (e["finish"] == 1)).astype(float)
    e["is1"] = (e["lane"] == 1).astype(float)
    e["mg"] = (e["course"] != e["lane"]).astype(float).where(e["course"].notna())
    e["a1"] = (e["racer_class"] == "A1").astype(float)
    e["nw"] = pd.to_numeric(e["nat_win"], errors="coerce")

    r = e.groupby(["date", "jcd", "rno", "場R"]).agg(
        in_win=("win1", "sum"),
        mg=("mg", "mean"),
        st_sd=("st", "std"),
        wind=("wind_speed", "max"),
        wave=("wave_height", "max"),
        a1=("a1", "mean"),
    ).reset_index()
    r["wakunari"] = (r["mg"] == 0).astype(float)
    # 1号艇の地力 − 他5艇平均
    p1 = e[e.lane == 1].groupby(["date", "jcd", "rno"])["nw"].mean()
    po = e[e.lane != 1].groupby(["date", "jcd", "rno"])["nw"].mean()
    gap = (p1 - po).rename("pow_gap").reset_index()
    r = r.merge(gap, on=["date", "jcd", "rno"], how="left")

    S = r.groupby("場R").agg(
        races=("in_win", "size"), in_win=("in_win", "mean"), maegumi=("mg", "mean"),
        wakunari=("wakunari", "mean"), st_sd=("st_sd", "mean"),
        wind=("wind", "mean"), wave=("wave", "mean"), a1=("a1", "mean"),
        pow_gap=("pow_gap", "mean")).reset_index()
    return S


def main():
    C = pd.read_csv("data/over100.csv")
    C = C[C["断面"] == "場×R"].copy()
    print(f"場×R のセル {len(C):,}  ({C['セグ'].nunique()} 場R × 券種目)")

    ok = C["前半的中"] >= MIN_HIT
    pool = C[ok & (C["前半回収"] > 1.0)].dropna(subset=["後半回収"]).copy()
    pool["生存"] = (pool["後半回収"] > 1.0).astype(int)
    print(f"前半100%超(的中{MIN_HIT}本以上): {len(pool):,}本 → "
          f"後半も超え {pool['生存'].sum():,}本 ({pool['生存'].mean():.1%})")

    S = structure()
    P = pool.merge(S, left_on="セグ", right_on="場R", how="left")
    print(f"構造を重ねられた: {P['場R'].notna().sum():,} / {len(P):,}")

    VARS = ["in_win", "maegumi", "wakunari", "st_sd", "wind", "wave", "a1", "pow_gap"]
    print(f"\n=== ① 構造は「後半も生きるか」を当てられるか ===")
    print(f"  {'変数':<10}{'生存セルの平均':>14}{'落ちたセルの平均':>16}{'差のt':>8}")
    for v in VARS:
        a = P.loc[P["生存"] == 1, v].dropna()
        b = P.loc[P["生存"] == 0, v].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(f"  {v:<10}{a.mean():>14.4f}{b.mean():>16.4f}{t:>8.2f}")
    print(f"  ※ |t|>2 が1本も無かったら、構造は黒箱の生き残りを当てられんいうこと")

    print(f"\n=== ② 目の形は効くか(1着艇番・荒れ度) ===")
    P["先頭"] = P["目"].str.split("-").str[0]
    P["平均艇番"] = P["目"].apply(
        lambda s: np.mean([int(x) for x in str(s).split("-")]))
    g = P.groupby("先頭").agg(n=("生存", "size"), keep=("生存", "mean"),
                              late=("後半回収", "mean"))
    print(f"  {'1着艇番':<8}{'本数':>7}{'生存率':>9}{'後半平均回収':>12}")
    for k, r in g.iterrows():
        print(f"  {k:<8}{int(r['n']):>7,}{r['keep']:>9.1%}{r['late']:>12.1%}")
    P["荒れ"] = pd.cut(P["平均艇番"], [0, 2.5, 3.5, 4.5, 7],
                      labels=["堅い", "中", "荒れ", "大荒れ"])
    g2 = P.groupby("荒れ", observed=True).agg(n=("生存", "size"),
                                              keep=("生存", "mean"),
                                              late=("後半回収", "mean"))
    print(f"\n  {'荒れ度':<8}{'本数':>7}{'生存率':>9}{'後半平均回収':>12}")
    for k, r in g2.iterrows():
        print(f"  {str(k):<8}{int(r['n']):>7,}{r['keep']:>9.1%}{r['late']:>12.1%}")

    print(f"\n=== ③ 生き残りは特定の場Rに固まっとるか ===")
    cnt = P.groupby("セグ")["生存"].agg(["size", "sum"])
    cnt["率"] = cnt["sum"] / cnt["size"]
    base = P["生存"].mean()
    # 帰無: 生存ラベルをシャッフルして、1場Rの最大生存数がどこまで行くか
    mx = []
    for _ in range(2000):
        sh = RNG.permutation(P["生存"].to_numpy())
        t = pd.Series(sh).groupby(P["セグ"].to_numpy()).sum()
        mx.append(t.max())
    thr = np.percentile(mx, 95)
    top = cnt.sort_values("sum", ascending=False).head(10)
    print(f"  全体の生存率 {base:.1%}。帰無での1場R最大生存数の95%点 = {thr:.0f}本")
    print(f"  {'場R':<12}{'前半通過':>9}{'後半も生存':>11}{'率':>8}")
    for k, r in top.iterrows():
        mark = " ← 閾値超え" if r["sum"] > thr else ""
        print(f"  {k:<12}{int(r['size']):>9}{int(r['sum']):>11}{r['率']:>8.1%}{mark}")

    print(f"\n=== ④ 構造を全部使って後半回収を予測できるか ===")
    d = P.dropna(subset=VARS + ["後半回収"])
    X = np.column_stack([np.ones(len(d))] + [d[v].to_numpy() for v in VARS])
    y = d["後半回収"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  n={len(d):,}  R² = {r2:.4f}")
    print(f"  ※ R²がほぼ0なら、構造は後半の回収を1ミリも説明せん")
    P.to_csv("data/overlay_mechanism.csv", index=False, encoding="utf-8-sig")
    print("\n保存: data/overlay_mechanism.csv")


if __name__ == "__main__":
    main()
