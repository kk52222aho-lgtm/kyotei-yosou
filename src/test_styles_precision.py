"""3連単20点/60点の"精度"＝どれだけ信用できる/どれだけ荒れるか（fat-tail精査）。

ユーザー: 「3連単20と60が一番勝つやん。その精度を知りたい」。
一番勝つに見える＝一番fat-tail(448/798本の万舟が作った)。4本立てで精度を出す:
  ①bootstrap CI(ROIの下限・100%割れ確率) ②万舟抜き耐性(上位N本除外)
  ③年別ROI(1年の偶然でないか) ④最大ドローダウン/最長連敗(万舟が来るまでの沈み=実戦の精度)
比較に 標準11点(頑健)も併記。1点100円・確定払戻・妙味(本命≠1号)・クリーン年。

例: python -u -m src.test_styles_precision
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import storage
from .predict import harville_trifecta
from .strategy_search import trifecta_probs
from .validate import fit_predict_leakfree, SELECTION_YEARS

BOOT = 3000


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.trio_combo, p.trio_yen,
               p.exacta_combo, p.exacta_yen, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def trio_rank(p):
    agg = defaultdict(float)
    for c, pr in trifecta_probs(p):
        agg["-".join(sorted(c.split("-"), key=int))] += pr
    return [c for c, _ in sorted(agg.items(), key=lambda x: -x[1])]


# スタイル: name, stake, ret関数
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
            if int(top["lane"]) == 1 or pd.isna(top["trifecta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            ex = []
            for i in p:
                di = 1 - p[i]
                if di > 0:
                    for j in p:
                        if j != i:
                            ex.append((f"{i}-{j}", p[i] * p[j] / di))
            ex.sort(key=lambda x: -x[1])
            ex_r = [c for c, _ in ex]
            tr_r = trio_rank(p)
            tf_r = [c for c, _ in sorted(harville_trifecta(p).items(), key=lambda x: -x[1])]
            won_t = (not pd.isna(top["tansho_lane"])) and int(top["tansho_lane"]) == int(top["lane"])
            tan = float(top["tansho_yen"]) if not pd.isna(top["tansho_yen"]) else 0.0
            exy = float(top["exacta_yen"]) if not pd.isna(top["exacta_yen"]) else 0.0
            tvy = float(top["trio_yen"])
            tfy = float(top["trifecta_yen"])
            exc, trc, tfc = str(top["exacta_combo"]), str(top["trio_combo"]), str(top["trifecta_combo"])
            # 各スタイルのreturn
            r11 = ((tan if won_t else 0) + (exy if exc in ex_r[:3] else 0)
                   + (tvy if trc in tr_r[:4] else 0) + (tfy if tfc in tf_r[:3] else 0))
            r20 = tfy if tfc in tf_r[:20] else 0
            r60 = tfy if tfc in tf_r[:60] else 0
            rows.append({"date": date, "yr": date[:4], "r11": r11, "r20": r20, "r60": r60})
    return rows


def precision(rows, key, stake, seed):
    rets = np.array([r[key] for r in rows])
    n = len(rets)
    tot_stake = n * stake
    roi = rets.sum() / tot_stake
    # bootstrap
    rng = np.random.default_rng(seed)
    boot = np.array([rets[rng.integers(0, n, n)].sum() / tot_stake for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sub = (boot < 1).mean()
    # 万舟抜き耐性
    srt = np.sort(rets)[::-1]
    rmv = {k: (srt[k:].sum()) / tot_stake for k in [0, 1, 5, 20, 50]}
    # 勝ちレース率(P&L>0) / 的中率(ret>0)
    pl = rets - stake
    win_rate = (pl > 0).mean()
    hit_rate = (rets > 0).mean()
    # 最大ドローダウン & 最長連敗(P&L<0連続)
    cum = np.cumsum(pl)
    peak = np.maximum.accumulate(cum)
    dd = int((peak - cum).max())
    streak = mx = 0
    for x in pl:
        streak = streak + 1 if x < 0 else 0
        mx = max(mx, streak)
    return {"roi": roi, "lo": lo, "hi": hi, "sub": sub, "rmv": rmv,
            "win": win_rate, "hit": hit_rate, "dd": dd, "streak": mx, "n": n}


def main():
    df = load()
    rows = collect(df)
    print(f"妙味(本命≠1号)・クリーン年 {len(rows):,}レース・1点100円\n")
    styles = [("標準11点", "r11", 1100), ("3連単20点", "r20", 2000), ("3連単60点", "r60", 6000)]
    for i, (name, key, stake) in enumerate(styles):
        s = precision(rows, key, stake, seed=i)
        print(f"■ {name}（1レース{stake:,}円）")
        print(f"   回収率 {s['roi']*100:.0f}%  bootstrap95%CI [{s['lo']*100:.0f}% , {s['hi']*100:.0f}%]  100%割れ確率 {s['sub']*100:.1f}%")
        print(f"   万舟抜き: " + " ".join(f"{k}本→{v*100:.0f}%" for k, v in s['rmv'].items()))
        yb = defaultdict(lambda: [0.0, 0])
        for r in rows:
            yb[r['yr']][0] += r[key]; yb[r['yr']][1] += stake
        yr_s = " ".join(f"{y}:{v[0]/v[1]*100:.0f}%" for y, v in sorted(yb.items()))
        print(f"   年別ROI: {yr_s}")
        print(f"   勝ちレース率(収支+) {s['win']*100:.0f}%  的中率(払戻>0) {s['hit']*100:.0f}%")
        print(f"   最大ドローダウン {s['dd']:,}円  最長連敗 {s['streak']}レース（＝万舟が来るまでの沈み）\n")
    print("※読み: CI下限が>100%かつ万舟50本抜いても>100%かつ年別全部>100%なら精度高い。")
    print("  ドローダウン/連敗が大きいほど、机上ROIは本物でも実戦は『耐える精度』が要る（メンタル/資金）。")


if __name__ == "__main__":
    main()
