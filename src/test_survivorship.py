"""コアエッジ(妙味=本命≠1号)がサバイバーシップの罠かを割る＝最重要の内部監査。

発見: win IS NOT NULL フィルタが、1号艇that転覆/失格/エンスト/不完走で着順なし→データ落ち
のレースを作る。するとモデルの本命that自動で非1号=「妙味」に誤分類される。これは
"モデルthaインを嫌った"でなく"1号thaレース中に飛んで消えた"=レース後情報の混入(サバイバーシップ)。
1号飛び=高配当ゆえROIを水増しし、ライブ(締切時1号は存在)で再現不能。
∴コア妙味を lane1あり(6艇そろい) vs lane1なし(飛んだ) で割り、あり側だけでエッジが
生き残るか測る。あり側>150%なら本物・罠は限定的。あり側that崩れれば全土台that水増しで確定。
確定払戻・walk-forward・リークなし(の検査そのもの)。

例: python -u -m src.test_survivorship
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
            has1 = bool((g["lane"] == 1).any())      # 1号艇thaデータに居るか(飛んでない)
            nboat = int(len(g))
            second = [int(l) for l in g["lane"] if int(l) != honmei][:3]
            hit_ex = str(top["exacta_combo"]) in {f"{honmei}-{s}" for s in second}
            tan_hit = (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == honmei
            rows.append({"yr": y, "has1": has1, "nboat": nboat,
                         "ex_ret": float(top["exacta_yen"]) if hit_ex else 0.0, "ex_hit": hit_ex,
                         "tan_ret": float(top["tansho_yen"]) if tan_hit else 0.0, "tan_hit": tan_hit})
    return pd.DataFrame(rows)


def line(sub):
    n = len(sub)
    if not n:
        return "0件"
    tan = sub["tan_ret"].sum() / (n * 100) * 100
    ex = sub["ex_ret"].sum() / (n * 3 * 100) * 100
    return f"{n:5d}件  単勝{tan:4.0f}%  2連単3点{ex:4.0f}%"


def main():
    df = load()
    d = collect(df)
    n = len(d)
    absent = int((~d["has1"]).sum())
    print(f"本命≠1号(妙味) 総数 {n:,}")
    print(f"  うち 1号艇that飛んで消えたレース(lane1なし) = {absent:,} ({absent/n*100:.0f}%)  <- ここが問題の核\n")

    print("[サバイバーシップ分解]")
    print(f"  1号あり(6艇そろい・真の妙味): {line(d[d['has1']])}")
    print(f"  1号なし(飛んで消えた)       : {line(d[~d['has1']])}")

    print("\n[出走数別=何艇消えてるか]")
    for nb in [6, 5, 4, 3]:
        sub = d[d["nboat"] == nb]
        if len(sub):
            print(f"  {nb}艇: {line(sub)}")

    print("\n[1号ありだけ・年別=真に張れる妙味の実力]")
    only = d[d["has1"]]
    for y in sorted(only["yr"].unique()):
        print(f"  {y}: {line(only[only['yr'] == y])}")

    print("\n判定: 1号あり側that>150%で3年一貫なら土台は本物・罠は限定的。")
    print("      1号あり側that100%付近まで落ちれば既存116/188%は飛び艇survivorshipの水増しで確定=最重要。")
    print("※注意: 欠場は事前公表で正当だが、転覆/失格はレース後=バックテストのみで起きlive再現不能。")


if __name__ == "__main__":
    main()
