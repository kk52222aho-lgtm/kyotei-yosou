"""②ライブ(朝=展示前)の確信とBT(展示込み)の確信が同じ分布か=テーブルを朝に適用できるか。

near-lock表はBT(展示込み特徴)の確信で作った。ライブ朝スキャンは展示前→展示特徴that欠落→
確信that別分布の恐れ。∴各テスト年の予測を「展示を潰した入力」(tenji_time/tenji_st→NaN=朝の再現)
で出し直し: ①朝の確信that何%まで届くか(分布) ②朝の確信≥80%が本当に84%当てるか(較正照合)。
通れば表を朝に使える、通らねば表差し替え or 締切前再スキャン必須。walk-forward・6艇クリーン。

例: python -u -m src.test_nearlock_live_gap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane FROM entries e JOIN payouts p
        ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df, blind):
    """blind=True: テスト年の展示(tenji_time/tenji_st)をNaN化=朝スキャン再現。"""
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        dfm = df.copy()
        if blind:
            for c in ("tenji_time", "tenji_st"):
                if c in dfm.columns:
                    dfm.loc[te, c] = np.nan            # 展示前=欠落
        test = fit_predict_leakfree(dfm, tr, te)
        for _, g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            s = g["p"].sum()
            if s <= 0 or pd.isna(top["tansho_lane"]):
                continue
            rows.append({"conf": float(top["p"]) / s * 100,
                         "won": int(int(top["tansho_lane"]) == int(top["lane"]))})
    return pd.DataFrame(rows)


def summary(d, label):
    n = len(d)
    print(f"[{label}] 本命 N={n:,}  確信: 中央{d['conf'].median():.0f}% 最大{d['conf'].max():.0f}%")
    print("  確信帯別: ≥70 / ≥80 / ≥85 の 件数 と 実的中%")
    for lo in [70, 80, 85]:
        sub = d[d["conf"] >= lo]
        if len(sub) < 20:
            print(f"    ≥{lo}%: {len(sub)}件（<20=この帯ほぼ発火せず）")
        else:
            print(f"    ≥{lo}%: {len(sub):6d}件  実的中{sub['won'].mean()*100:.1f}%")


def main():
    df = load()
    print("=== ② 展示込み(BT物差し) vs 展示前(朝=ライブ) の確信分布・較正 ===\n")
    bt = collect(df, blind=False)
    summary(bt, "展示込み(=near-lock表を作った物差し)")
    print()
    live = collect(df, blind=True)
    summary(live, "展示前(=朝スキャンの再現)")
    print("\n判定: 朝が『≥80%帯that十分発火 & 実的中≈84%』なら表を朝に適用可。")
    print("      朝で≥80%that激減 or 的中that84%とズレるなら=分布が別物=表は朝に使えない")
    print("      →near-lockは締切前(展示込み)再スキャンでしか成立しない。")


if __name__ == "__main__":
    main()
