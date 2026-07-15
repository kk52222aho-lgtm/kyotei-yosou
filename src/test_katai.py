"""勝てる目(高的中)だけ出す=逆張り妙味を捨て、当たる本命を測る。

今日の確定: 予測構造は本物(1にイン#1、2に地力#2)、だがエッジは無い(市場that値付け済)。
∴儲けでなく"当たる"に振る。何that一番当たるか=イン×その選手that強い(地力上位)that最も堅いはず。
単勝(イン/モデル本命)の的中率を、地力・確信度で切って測る+2連単1流しの的中率。
儲けは主張しない(低配当)。的中率で"勝てる目"を出す土台。6艇クリーン・確定払戻・walk-forward。

例: python -u -m src.test_katai
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen, p.exacta_combo, p.exacta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
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
            rk = g["nat_win"].rank(ascending=False, method="min")
            top = g.iloc[0]
            win_lane = int(top["tansho_lane"])
            in1 = g[g["lane"] == 1]
            if in1.empty:
                continue
            in_pow_rank = int(rk[in1.index[0]])
            honmei = int(top["lane"])
            p = {int(l): float(v) for l, v in zip(g["lane"], g["p"])}
            s = sum(p.values()) or 1
            hon_pct = p[honmei] / s * 100
            second = [int(l) for l in g["lane"] if l != honmei][:3]
            ex_hit = str(top["exacta_combo"]) in {f"{honmei}-{x}" for x in second}
            rows.append({
                "yr": y, "in_pow_rank": in_pow_rank, "hon_pct": hon_pct, "honmei": honmei,
                "in_win": int(win_lane == 1),
                "in_ret": float(top["tansho_yen"]) if win_lane == 1 else 0.0,
                "hon_win": int(win_lane == honmei),
                "hon_ret": float(top["tansho_yen"]) if win_lane == honmei else 0.0,
                "ex_hit": int(ex_hit), "ex_ret": float(top["exacta_yen"]) if ex_hit else 0.0,
            })
    return pd.DataFrame(rows)


def stat(sub, win, ret, pts):
    n = len(sub)
    if not n:
        return "0件"
    return f"的中{sub[win].mean()*100:4.1f}%  回収{sub[ret].sum()/(n*pts*100)*100:4.0f}%  (n{n})"


def main():
    df = load()
    d = collect(df)
    print(f"6艇そろい 総数 {len(d):,}\n")

    print("[イン単勝(1号) 的中率 × インの地力ランク] 強い選手that内=最も堅い")
    for k in [1, 2, 3, 4, 5, 6]:
        sub = d[d["in_pow_rank"] == k]
        print(f"  インの地力{k}位: {stat(sub, 'in_win', 'in_ret', 1)}")

    print("\n[モデル本命(単勝) 的中率 × 本命の確信度(正規化勝率)]")
    for lo, hi, lb in [(0, 30, "〜30%"), (30, 40, "30-40%"), (40, 50, "40-50%"), (50, 100, "50%〜")]:
        sub = d[(d["hon_pct"] >= lo) & (d["hon_pct"] < hi)]
        print(f"  確信{lb:7s}: {stat(sub, 'hon_win', 'hon_ret', 1)}")

    print("\n[堅い狙い: 本命=1号(イン) かつ インが地力1-2位]")
    strong_in = d[(d["honmei"] == 1) & (d["in_pow_rank"] <= 2)]
    print(f"  単勝    : {stat(strong_in, 'hon_win', 'hon_ret', 1)}")
    print(f"  2連単3点: {stat(strong_in, 'ex_hit', 'ex_ret', 3)}")

    print("\n※的中率=当たる度。回収<100%は当たるthat控除で負ける(市場that値付け済)を意味する。")
    print("  勝てる目=高的中の堅い本命。儲けは主張しない。6艇クリーン・確定払戻・walk-forward。")


if __name__ == "__main__":
    main()
