"""segment sweepで跳ねた候補(メイン10-12R / B1本命)を敵対的に検証＝まぐれか本物か。

test_segmentsは5軸×4バケツ=約20スライスの釣り→20個切れば1-2個は偶然230%超える(多重比較)。
跳ねたメイン10-12R(255%)/B1本命(257%)を、飛びつかず3つの敵対検証にかける:
  1.年別3年で一貫か(1年のまぐれ除外) 2.fat-tail頑健か(上位配当抜いても残るか)
  3.基準171%と件数を踏まえ、多重比較を割り引いても生きるか
通れば新シナリオ候補、崩れれば釣りの蜃気楼で正直にクローズ。確定払戻・walk-forward。

例: python -u -m src.test_segments_verify
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.exacta_combo, p.exacta_yen
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
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            honmei = int(top["lane"])
            if honmei == 1 or pd.isna(top["exacta_yen"]):
                continue
            second = [int(l) for l in g["lane"] if int(l) != honmei][:3]
            hit = str(top["exacta_combo"]) in {f"{honmei}-{s}" for s in second}
            rows.append({"yr": y, "rno": int(rno), "cls": str(top["racer_class"]),
                         "hit": hit, "ret": float(top["exacta_yen"]) if hit else 0.0})
    return pd.DataFrame(rows)


def roi(sub):
    n = len(sub)
    return (sub["ret"].sum() / (n * 3 * 100) * 100) if n else 0


def fattail(sub):
    """上位配当を抜きながらROIの梯子=fat-tail依存度(分母は件数固定=1点あたり品質)。"""
    n = len(sub)
    rets = np.sort(sub["ret"].to_numpy())[::-1]
    out = []
    for k in [0, 3, 5, 10]:
        if n - k <= 0:
            out.append("--")
            continue
        r = rets[k:].sum() / (n * 3 * 100) * 100
        out.append(f"{r:.0f}%")
    return " / ".join(out)


def report(d, name, mask):
    sub = d[mask]
    n = len(sub)
    print(f"\n[{name}]  N={n}  全体ROI={roi(sub):.0f}%")
    print("  年別:", "  ".join(f"{y}:{roi(sub[sub['yr']==y]):.0f}%(n{len(sub[sub['yr']==y])})"
                              for y in sorted(sub["yr"].unique())))
    print(f"  fat-tail(上位0/3/5/10本抜き): {fattail(sub)}")


def main():
    df = load()
    d = collect(df)
    print(f"本命≠1号 総数 {len(d):,}  基準ROI {roi(d):.0f}%")
    report(d, "基準(全体)", pd.Series(True, index=d.index))
    report(d, "候補A メイン10-12R", d["rno"].between(10, 12))
    report(d, "候補B B1本命", d["cls"] == "B1")
    report(d, "候補A∩B メイン×B1", d["rno"].between(10, 12) & (d["cls"] == "B1"))
    print("\n判定基準: 3年とも>基準 かつ 上位5本抜いても>150% なら本物寄り。")
    print("        1年依存/上位数本で消える なら多重比較の蜃気楼=正直にクローズ。")
    print("※確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
