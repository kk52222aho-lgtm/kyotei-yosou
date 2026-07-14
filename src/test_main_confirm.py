"""メイン10-12R増幅(255%)を潰しにかかる=proxyでないか最終確認＋機序の裏取り。

敵対検証を通過(3年一様・fat-tail頑健)。残る疑い=(1)特定の場に偏ってへんか(場proxy)
(2)機序"インのA1スターthaが名前で過剰買い→崩せる時ほど深い"が本当なら、インがA1の時ほど妙味が深いはず。
確認: メイン10-12Rの本命≠1号を、1.イン(1号艇)のクラス別ROI 2.場をleave-one-out(最大寄与の場を
抜いても残るか)で割る。機序どおり(インA1で最大)かつ場分散なら確定→サイト実装候補。確定払戻・walk-forward。

例: python -u -m src.test_main_confirm
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


def collect_main(df):
    """メイン10-12Rの本命≠1号だけ集め、インのクラスと場も刻む。"""
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if not 10 <= int(rno) <= 12:
                continue
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            honmei = int(top["lane"])
            if honmei == 1 or pd.isna(top["exacta_yen"]):
                continue
            lane1 = g[g["lane"] == 1]
            in_cls = str(lane1.iloc[0]["racer_class"]) if not lane1.empty else "?"
            second = [int(l) for l in g["lane"] if int(l) != honmei][:3]
            hit = str(top["exacta_combo"]) in {f"{honmei}-{s}" for s in second}
            rows.append({"yr": y, "jcd": str(jcd), "in_cls": in_cls,
                         "hit": hit, "ret": float(top["exacta_yen"]) if hit else 0.0})
    return pd.DataFrame(rows)


def roi(sub):
    n = len(sub)
    return (sub["ret"].sum() / (n * 3 * 100) * 100) if n else 0


def main():
    df = load()
    d = collect_main(df)
    print(f"メイン10-12R 本命≠1号 総数 {len(d):,}  ROI {roi(d):.0f}%\n")

    print("[機序: イン(1号艇)のクラス別] スター(A1)ほど過剰買い→妙味深い、that機序")
    for c in ["A1", "A2", "B1", "B2"]:
        sub = d[d["in_cls"] == c]
        if len(sub) >= 30:
            print(f"  イン={c:3s}: {len(sub):4d}件  ROI{roi(sub):4.0f}%")
        else:
            print(f"  イン={c:3s}: {len(sub):4d}件  (小)")

    print("\n[場proxyチェック: 最大寄与の場をleave-one-out]")
    vc = d["jcd"].value_counts()
    print(f"  開催場数={d['jcd'].nunique()}  最多場={vc.index[0]}({vc.iloc[0]}件)")
    top_v = vc.index[0]
    print(f"  全体ROI{roi(d):.0f}% → 最多場{top_v}を除くと {roi(d[d['jcd']!=top_v]):.0f}% (残れば場proxyでない)")
    per = [(v, len(d[d['jcd'] == v]), roi(d[d['jcd'] == v])) for v in vc.index if len(d[d['jcd'] == v]) >= 40]
    hi = sum(1 for _, _, r in per if r >= 171)
    print(f"  件数40+の場 {len(per)}個中 {hi}個that基準171%超 (多数なら特定場でなく横断的)")

    print("\n判定: インA1で最大 かつ 最多場を抜いても>200% かつ 多くの場で基準超え なら")
    print("      メイン増幅=横断的な群衆バイアス(スター過剰買い)で確定→サイト実装候補。")
    print("※確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
