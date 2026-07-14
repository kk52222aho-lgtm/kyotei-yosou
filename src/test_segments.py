"""群衆that構造的に間違える"別の断面"を探す＝既存エッジを条件でスライスして増幅点を見つける。

今日の展開実験で確定: "予測に情報を足す"道は死(市場that公開情報を全部値付け)。
∴次は"もっと賢い予測"でなく"群衆that情報処理をサボる/間違える断面"を探す。
エッジ本体=群衆thaインを過剰に買う。問い=その過剰さが一番現実からズレるのはどの断面か。
既存の確定エッジ(本命≠1号の2連単3点=約183%)を条件でスライスし、増幅する断面を出す:
  風速/波高(条件=群衆that紙スペックに固執し織込みが遅いsticky prior・荒天ほどイン過剰買い?)
  レース番号(朝イチ/注目薄=薄い市場ほど歪む? メインはシャープ?)
  本命の枠(2差し vs 4まくり=群衆はまくり軽視?)
  本命クラス(格下を推すとき名前バイアスで妙味増?)
新特徴でなく既存エッジの断面。確定払戻・walk-forward・リークなし。

例: python -u -m src.test_segments
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.exacta_combo, p.exacta_yen, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.exacta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df):
    """クリーン年walk-forwardで本命≠1号レースを集め、条件と2連単3点の結果を刻む。"""
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
            buys = {f"{honmei}-{s}" for s in second}
            hit = str(top["exacta_combo"]) in buys
            rows.append({
                "honmei": honmei, "cls": str(top["racer_class"]),
                "wind": float(top["wind_speed"]) if not pd.isna(top["wind_speed"]) else -1,
                "wave": float(top["wave_height"]) if not pd.isna(top["wave_height"]) else -1,
                "rno": int(rno), "hit": hit, "ret": float(top["exacta_yen"]) if hit else 0.0,
            })
    return pd.DataFrame(rows)


def roi_line(sub):
    n = len(sub)
    if n < 40:
        return f"{n:5d}件  (小・判定不能)"
    roi = sub["ret"].sum() / (n * 3 * 100) * 100
    hit = sub["hit"].mean() * 100
    mark = "  <== 増幅" if roi >= 230 else ("  <-- 減衰" if roi < 150 else "")
    return f"{n:5d}件  的中{hit:4.0f}%  ROI{roi:4.0f}%{mark}"


def seg(df, name, buckets):
    print(f"\n[{name}]")
    for label, mask in buckets:
        print(f"  {label:16s}: {roi_line(df[mask])}")


def main():
    df = load()
    d = collect(df)
    base = d["ret"].sum() / (len(d) * 3 * 100) * 100
    print(f"本命≠1号レース総数 {len(d):,}  全体2連単3点ROI={base:.0f}% (これが基準=約183%系)")
    print("各断面でROIthaが基準を大きく超える所=群衆の過剰買いthaが一番デカい断面")

    seg(d, "風速(m/s) 荒天ほどイン崩れ→群衆sticky?", [
        ("穏 0-1", d["wind"].between(0, 1)), ("並 2-3", d["wind"].between(2, 3)),
        ("強 4-5", d["wind"].between(4, 5)), ("烈 6+", d["wind"] >= 6)])
    seg(d, "波高(cm) 高波ほどイン不利", [
        ("平水 0", d["wave"] == 0), ("小 1-2", d["wave"].between(1, 2)),
        ("中 3-4", d["wave"].between(3, 4)), ("高 5+", d["wave"] >= 5)])
    seg(d, "レース番号 朝イチ/注目薄=薄い市場?", [
        ("序盤 1-3R", d["rno"].between(1, 3)), ("中盤 4-6R", d["rno"].between(4, 6)),
        ("終盤 7-9R", d["rno"].between(7, 9)), ("メイン 10-12R", d["rno"].between(10, 12))])
    seg(d, "本命の枠 群衆はまくり(4-6)軽視?", [
        ("2号(差し)", d["honmei"] == 2), ("3号", d["honmei"] == 3),
        ("4号(まくり)", d["honmei"] == 4), ("5-6号(大外)", d["honmei"] >= 5)])
    seg(d, "本命クラス 格下推し=名前バイアス妙味?", [
        ("A1", d["cls"] == "A1"), ("A2", d["cls"] == "A2"),
        ("B1", d["cls"] == "B1"), ("B2", d["cls"] == "B2")])

    print("\n※基準183%を大きく超える断面=そこで群衆の過剰買いthaより深い=新シナリオ候補。")
    print("  確定払戻・walk-forward・リークなし。小標本(40未満)は判定不能。次段=生き残り断面をfat-tail頑健性へ。")


if __name__ == "__main__":
    main()
