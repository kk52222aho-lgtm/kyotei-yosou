"""「単勝/2連単が死ぬ群でも3連複だけ生き残るレース」をあぶり出す。

現状の見送り群: ①本命=1号艇(インpicks・妙味なし) ②本命の堅さ(chalk)が高い群。
問い: そこで単勝/2連単は控除負けでも、3連複(着順不問の器)だけROI>100%が残るか？
  残るなら"3連複のみ買い"の新セグメントとしてサイトにあぶり出す価値がある。

規律: 確定払戻ベース・クリーン年walk-forward・全レース。単勝は本命フラット(勝ち→tansho_yen)。
  chalkの代理は市場<1.5でなく【モデル本命の正規化勝率 p0】(leak-free・全レースで常に取れる)。
  p0高=モデル堅い≈単勝低め。真の単勝<1.5は勝ち馬しかオッズ確定できず全群に当てられんため。
判定: 単勝<100 かつ 2連単<100 かつ 3連複>=100 のセグメント = 3連複のみ生存 = あぶり出し対象。

例: python -u -m src.test_trio_only
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import storage
from .strategy_search import trifecta_probs
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.trio_combo, p.trio_yen,
               p.exacta_combo, p.exacta_yen, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trio_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def trio_probs(p):
    agg = defaultdict(float)
    for combo, prob in trifecta_probs(p):
        agg["-".join(sorted(combo.split("-"), key=int))] += prob
    return sorted(agg.items(), key=lambda x: -x[1])


def collect(df):
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for _, g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if pd.isna(top["trio_yen"]) or pd.isna(top["trifecta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            honmei = int(top["lane"])
            p0 = p[honmei]
            # 単勝(本命フラット)
            won_t = (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == honmei
            tan_ret = float(top["tansho_yen"]) if won_t else 0.0
            # 2連単 上位3点
            ex = []
            for i in p:
                di = 1 - p[i]
                if di > 0:
                    for j in p:
                        if j != i:
                            ex.append((f"{i}-{j}", p[i] * p[j] / di))
            ex.sort(key=lambda x: -x[1])
            ex_pick = [c for c, _ in ex[:3]]
            ex_hit = str(top["exacta_combo"]) in ex_pick
            ex_ret = float(top["exacta_yen"]) if (ex_hit and not pd.isna(top["exacta_yen"])) else 0.0
            # 3連複 上位4点
            tr_pick = [c for c, _ in trio_probs(p)[:4]]
            tr_hit = str(top["trio_combo"]) in tr_pick
            tr_ret = float(top["trio_yen"]) if tr_hit else 0.0
            rows.append({
                "contrarian": honmei != 1, "p0": p0,
                "tan_ret": tan_ret, "ex_ret": ex_ret, "tr_ret": tr_ret,
            })
    return rows


def seg_roi(rows):
    n = len(rows)
    if not n:
        return None
    tan = sum(r["tan_ret"] for r in rows) / (n * 100)
    ex = sum(r["ex_ret"] for r in rows) / (n * 3 * 100)
    tr = sum(r["tr_ret"] for r in rows) / (n * 4 * 100)
    return n, tan, ex, tr


def main():
    df = load()
    rows = collect(df)
    print(f"クリーン年 全 {len(rows):,} レース  [単勝1点 / 2連単3点 / 3連複4点 の回収率]\n")

    # セグメント: 本命=1号艇か × p0(モデル堅さ=chalk代理)帯
    def label(r):
        side = "本命≠1号(妙味)" if r["contrarian"] else "本命=1号(イン)"
        if r["p0"] >= 0.50:
            band = "堅い(p0≥.50)"
        elif r["p0"] >= 0.35:
            band = "中(.35-.50)"
        else:
            band = "薄い(p0<.35)"
        return f"{side}/{band}"

    segs = defaultdict(list)
    for r in rows:
        segs[label(r)].append(r)

    print(f"  {'セグメント':<26}{'N':>6}{'単勝':>8}{'2連単':>8}{'3連複':>8}   3連複のみ生存?")
    survivors = []
    for lab in sorted(segs):
        st = seg_roi(segs[lab])
        if not st or st[0] < 50:
            if st:
                print(f"  {lab:<26}{st[0]:>6}{'—(薄)':>8}")
            continue
        n, tan, ex, tr = st
        only = tan < 1.0 and ex < 1.0 and tr >= 1.0
        mark = "★あぶり出し" if only else ("(全部生存)" if tan >= 1 and ex >= 1 and tr >= 1 else "")
        if only:
            survivors.append((lab, n, tan, ex, tr))
        print(f"  {lab:<26}{n:>6}{tan*100:>7.0f}%{ex*100:>7.0f}%{tr*100:>7.0f}%   {mark}")

    print("\n--- あぶり出し結論 ---")
    if survivors:
        print("  単勝<100 & 2連単<100 だが 3連複≥100 のセグメント（＝3連複のみ買いが妥当）:")
        for lab, n, tan, ex, tr in survivors:
            print(f"    ・{lab}: N={n} 単勝{tan*100:.0f}%/2連単{ex*100:.0f}%/3連複{tr*100:.0f}%")
        print("  → このセグメントを『3連複のみ残す』としてサイトにあぶり出す価値あり（確定払戻ベース）。")
    else:
        print("  該当なし＝単勝/2連単が死ぬ群では3連複も生存してない。")
        print("  → 3連複は独立の器でなく、妙味レース選択にぶら下がってる（既知の結論と整合）。")
        print("     『3連複だけ残す』は成立せず＝現状の妙味ゲートのまま出すのが正しい。")
    print("\n※chalk代理はp0(モデル堅さ)。真の単勝<1.5は勝ち馬しかオッズ確定できず全群に当てられん。")
    print("※確定払戻ベース＝ライブ未証明。単勝/2連単の低ROIは既知(妙味外は控除負け)。")


if __name__ == "__main__":
    main()
