"""EV-ranking プローブ: 締切オッズはdecision時にactionableか(=EVベットがフラットを超えるか)。

前提: 土台116/188(確定払戻)で「contrarianが控除後に市場を超える」は既に確定。
  残る問い=締切オッズをdecision時に使い「賭け方を良く」できるか(EVベット)。
問い: EV=model_p×締切odds でbetを並べ替えたとき、高EVバケットが低EVより高ROIか。
  YES→締切オッズはactionable=前向き配管(数ヶ月のライブ締切収集)に投資する根拠。
  NO →オッズはEVを足さん=配管は無駄(keirinのEV無エッジと同じ)。
規律:
  - ROIは確定払戻(絶対値)。EV計算の締切オッズは29%乖離で汚れる→信頼オッズ部分集合で。
  - contrarian clean年・walk-forward。フラット(全張り)ROIをbaselineに刻む。
  - バケットの薄い所は件数併記で保留。これは歴史スナップショットの一なめ=方向を見るだけ。

例: python -u -m src.test_ev_probe
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import storage
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS

TAN = os.path.join(storage.DATA_DIR, "odds_cache.json")
EX = os.path.join(storage.DATA_DIR, "exacta_odds_cache.json")


def collect(df):
    tan = json.load(open(TAN, encoding="utf-8")) if os.path.exists(TAN) else {}
    ex_c = json.load(open(EX, encoding="utf-8")) if os.path.exists(EX) else {}
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
            if int(top["lane"]) == 1 or pd.isna(top["tansho_yen"]):
                continue
            key = f"{date}-{jcd}-{rno}"
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            honmei = int(top["lane"])
            won_tan = int(top["tansho_lane"]) == honmei
            # 本命の単勝締切オッズ: 勝ち→払戻/100、負け→キャッシュ
            if won_tan:
                tan_odds = top["tansho_yen"] / 100.0
            else:
                od = tan.get(key)
                tan_odds = od.get(str(honmei)) if od else None
            if tan_odds is None:
                continue
            # 信頼: 勝ち組(単勝)の締切×100≈確定
            trust = abs((top["tansho_yen"] / 100.0) * 100 - top["tansho_yen"]) < 1 if won_tan else None
            # 2連単 top3 と EV
            ex = ex_c.get(key)
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            pmap = dict(exacta_probs(p))
            ex_ev = ex_win = ex_pay = None
            if ex and len(ex) >= 30 and all(ex.get(c) for c in top3):
                ex_ev = sum(pmap[c] * ex[c] for c in top3) / 3.0   # 3点均等のEV比
                ex_win = top["exacta_combo"] in top3
                ex_pay = int(top["exacta_yen"]) if not pd.isna(top["exacta_yen"]) else 0
                # 単勝信頼を勝ち組2連単オッズでも定義（負けレースの信頼判定用）
                wc = top["exacta_combo"]; wy = top["exacta_yen"]
                if not pd.isna(wy) and ex.get(wc):
                    trust = abs(ex[wc] * 100 - wy) / wy <= 0.10
            rows.append({
                "ex_ev": ex_ev, "ex_win": ex_win,
                "ex_ret": (ex_pay if ex_win else 0) if ex_pay is not None else None,  # ★的中時のみ払戻
                "trust": trust,
            })
    return pd.DataFrame(rows)


def roi_by_bucket(d, ev_col, win_col, ret_col, stake, label, edges):
    """ret_col=的中時のみ払戻(ゲート済)。ROI=Σret/(N×stake)。"""
    print(f"\n=== {label}: EVバケット別ROI（フラット全張りがbaseline）===")
    sub = d.dropna(subset=[ev_col, ret_col])
    flat = sub[ret_col].sum() / (len(sub) * stake)
    print(f"  フラット全張り: ROI {flat*100:.1f}%（{len(sub):,}ベット）")
    print(f"  {'EV帯':<12}{'ベット':>7}{'的中率':>8}{'ROI':>8}{'上位1的中への依存':>16}")
    lbls = [f"<{edges[1]}"] + [f"{edges[i]}-{edges[i+1]}" for i in range(1, len(edges) - 2)] + [f">{edges[-2]}"]
    for i in range(len(edges) - 1):
        b = sub[(sub[ev_col] >= edges[i]) & (sub[ev_col] < edges[i + 1])]
        if len(b) == 0:
            continue
        tot = b[ret_col].sum()
        roi = tot / (len(b) * stake)
        # fat-tail依存: 最大払戻1本を抜くとROIどうなるか
        top1 = b[ret_col].max()
        roi_x1 = (tot - top1) / (len(b) * stake) if len(b) else 0
        hold = " ※薄" if len(b) < 50 else ""
        print(f"  {lbls[i]:<12}{len(b):>7,}{b[win_col].mean()*100:>7.1f}%{roi*100:>7.1f}%"
              f"{'  最大1本抜き'+f'{roi_x1*100:.0f}%':>16}{hold}")


def main():
    df = load()
    d = collect(df)
    print(f"contrarian clean年・2連単EVあり {d.ex_ev.notna().sum():,} / 信頼 {int((d.trust==True).sum()):,}")
    print("※単勝EVは勝ち馬の締切オッズがキャッシュに無く(確定払戻で代用=リーク)、leak-free不可→前向き収集待ちで今日は除外")

    for name, dd in [("全レース", d), ("信頼オッズのみ", d[d.trust == True])]:
        exd = dd.dropna(subset=["ex_ev"])
        print("\n" + "#" * 60 + f"\n### 2連単3点 [{name}]（{len(exd):,}）")
        roi_by_bucket(exd, "ex_ev", "ex_win", "ex_ret", 300, name,
                      [0, 0.8, 1.0, 1.3, 2.0, 1e9])
    # 配管GO/NO-GOの核心: 信頼>2.0帯のfat-tail頑健性
    hb = d[(d.trust == True) & (d.ex_ev >= 2.0)].dropna(subset=["ex_ret"])
    rets = np.sort(hb["ex_ret"].to_numpy())[::-1]
    n = len(hb); stake = n * 300
    print(f"\n=== 【配管判定】信頼>2.0帯 {n}ベットのfat-tail頑健性 ===")
    print(f"  的中 {int((rets>0).sum())}本 / 最大払戻 {int(rets[0]):,}円")
    for k in [0, 1, 5, 10, 20, 40]:
        print(f"  上位{k:>2}本除外: ROI {rets[k:].sum()/stake*100:>6.0f}%")
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(rets, n, replace=True).sum() / stake for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap 95%CI [{lo*100:.0f}% , {hi*100:.0f}%] / 100%割れ確率 {(boot<1).mean()*100:.0f}%")
    print("\n※上位20-40本除いても>150%かつCI下限>100%なら=本物のactionableエッジ=配管GO。")
    print("※上位数本除くと100%近辺に崩れる/CI下限<100%=fat-tail蜃気楼=配管NO-GO(keirin的結末)。")
    print("※EVはleak-free(締切キャッシュ)、ROIは確定払戻(的中ゲート済)。ただし>2.0帯は高オッズ=29%乖離が最も効く帯。")


if __name__ == "__main__":
    main()
