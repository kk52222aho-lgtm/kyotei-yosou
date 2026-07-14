"""「2番目の答え」= イン(#1)that負けた時、勝者を決めてるのは何か。

#1=イン55%勝ちのmetric内で#2を足しても埋もれる(全部~0)。∴切り方を変える:
インthaが勝たなかったレース(=位置の下駄that消えた世界)だけに絞り、非イン勝者を
当てる力を選抜器ごとに比較する。一番当てる軸=位置that崩れた時の王様=真の第二の答え。
6艇そろい(ライブ再現可)・確定払戻・walk-forward・リークなし。

例: python -u -m src.test_when_inside_fails
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def main():
    df = load()
    df["_st"] = df["tenji_st"].fillna(df["avg_st"])
    sel = {
        "モデル(全部)": None,
        "地力(全国勝率)": "nat_win",
        "展示タイム(速さ)": "tenji_time",
        "出足(相対ST・小さい程速)": "_st",
        "モーター2連率": "motor_2rate",
        "当地勝率": "loc_win",
    }
    tot = 0
    hit = {k: 0 for k in sel}
    hit["2号ベタ(差し定番)"] = 0
    hit["外(4-6)最速ST"] = 0

    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6 or not (g["lane"] == 1).any():
                continue
            win_lane = int(g.iloc[0]["tansho_lane"])
            if win_lane == 1:            # インが勝ったレースは除外(=#1の世界)
                continue
            tot += 1
            non1 = g[g["lane"] != 1]
            for name, col in sel.items():
                if col is None:
                    pick = int(non1.sort_values("p", ascending=False).iloc[0]["lane"])
                elif col in ("_st", "tenji_time"):
                    pick = int(non1.sort_values(col, ascending=True).iloc[0]["lane"])
                else:
                    pick = int(non1.sort_values(col, ascending=False).iloc[0]["lane"])
                if pick == win_lane:
                    hit[name] += 1
            if win_lane == 2:
                hit["2号ベタ(差し定番)"] += 1
            outer = non1[non1["lane"] >= 4]
            if not outer.empty and int(outer.sort_values("_st", ascending=True).iloc[0]["lane"]) == win_lane:
                hit["外(4-6)最速ST"] += 1

    print(f"インthaが負けたレース(6艇そろい) {tot:,}件  ← この世界で勝者を当てる力\n")
    order = sorted(hit.items(), key=lambda x: -x[1])
    for name, h in order:
        print(f"  {name:22s}: 的中{h/tot*100:5.1f}%  ({h}件)")
    print("\n読み: 一番当てる軸that『イン(#1)that崩れた時の王様』=2番目の答え。")
    print("      2号ベタ/モデルを明確に超える軸that出れば、それが位置の次に効く本物の力。")
    print("※6艇そろい・確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
