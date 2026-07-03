"""①フィルタ閾値のOOF検証：1.5はin-sampleだったか、凍結すると落ち幅は？

答える問い（これだけ）: 単勝<T 見送り閾値を「評価年(23-25)を見ずに」選び直し、
  23-25へ凍結適用したときの2連単ROI。in-sample 1.5(=208.3%)からの落ち幅を見る。
答えない問い: 戦略全体が{2022,2026}にオーバーフィットしてへんか（①では不可）。

隔離設計:
  - 閾値選択は {2022,2026} のみ。しかもクリーン年のモデルを通さない:
      2022 ← 2026で学習して採点 / 2026 ← 2022で学習して採点
  - 2022単独・2026単独の最適閾値も出す（一致すれば頑健、バラければ1.5は薄い母数の産物）
  - 評価(23-25)はwalk-forward(train=それ以前)＋凍結閾値

出力（3点）:
  1. {2022,2026}で選んだ凍結閾値 ＋ 2022単独/2026単独の最適値
  2. 23-25に凍結適用した真OOFの2連単ROI（＋単勝・合計）
  3. 188.7(無フィルタOOF=T1.0)と 208.3(in-sample T1.5)への差分

オッズは data/odds_cache.json にキャッシュ（本命の確定単勝オッズ、勝ちは払戻から）。
例: python -u -m src.test_filter_oof
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import scraper, storage
from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree

CACHE = os.path.join(storage.DATA_DIR, "odds_cache.json")
GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 15.0]


def _load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {}


def collect(df, test_year, train_mask):
    te = (df["yr"] == test_year).to_numpy()
    test = fit_predict_leakfree(df, train_mask, te)
    rows = []
    for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        top = g.iloc[0]
        if int(top["lane"]) == 1 or pd.isna(top["tansho_lane"]):
            continue
        s = g["p"].sum()
        top3 = []
        if s > 0 and not pd.isna(top["exacta_yen"]):
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            top3 = [c for c, _ in exacta_probs(p)[:3]]
        won = int(top["tansho_lane"]) == int(top["lane"])
        rows.append({
            "key": f"{date}-{jcd}-{rno}", "date": date, "jcd": jcd, "rno": int(rno),
            "honmei": int(top["lane"]), "won": won, "tansho_yen": int(top["tansho_yen"]),
            "top3": top3,
            "exacta_combo": top["exacta_combo"],
            "exacta_yen": None if pd.isna(top["exacta_yen"]) else int(top["exacta_yen"]),
        })
    return rows


def resolve_odds(rows, cache):
    """本命の単勝オッズ: 勝ち→払戻/100、負け→キャッシュ or スクレイプ。"""
    n_scr = 0
    for r in rows:
        if r["won"]:
            r["odds"] = r["tansho_yen"] / 100.0
            continue
        ck = r["key"]
        if ck not in cache:
            od = scraper.fetch_odds(r["date"], r["jcd"], r["rno"])
            # fetch_odds は int キー → str に正規化して保存（JSONも str キー）
            cache[ck] = {str(k): v for k, v in od.items()} if od else {}
            n_scr += 1
            time.sleep(0.3)
            if n_scr % 100 == 0:
                json.dump(cache, open(CACHE, "w", encoding="utf-8"))
                print(f"  scraped {n_scr}")
        od = cache[ck]
        r["odds"] = od.get(str(r["honmei"])) if od else None
    return n_scr


def roi(rows, T, kind):
    kept = [r for r in rows if not (r["odds"] is not None and r["odds"] < T)]
    if kind == "exacta":
        er = [r for r in kept if r["top3"] and r["exacta_yen"] is not None]
        if not er:
            return None, 0
        ret = sum(r["exacta_yen"] for r in er if r["exacta_combo"] in r["top3"])
        hits = sum(1 for r in er if r["exacta_combo"] in r["top3"])
        return ret / (len(er) * 300), hits
    if kind == "tansho":
        ret = sum(r["tansho_yen"] for r in kept if r["won"])
        return ret / (len(kept) * 100), sum(1 for r in kept if r["won"])
    # combined
    er = [r for r in kept if r["top3"] and r["exacta_yen"] is not None]
    stake = len(kept) * 100 + len(er) * 300
    ret = (sum(r["tansho_yen"] for r in kept if r["won"])
           + sum(r["exacta_yen"] for r in er if r["exacta_combo"] in r["top3"]))
    return ret / stake if stake else None, None


def best_T(rows):
    scored = [(T, roi(rows, T, "exacta")[0]) for T in GRID]
    scored = [(T, v) for T, v in scored if v is not None]
    return max(scored, key=lambda x: x[1])


def main():
    df = load()
    cache = _load_cache()

    print("収集: 2022(←2026学習)/2026(←2022学習)/2023-25(walk-forward)")
    sel_2022 = collect(df, "2022", (df["yr"] == "2026").to_numpy())
    sel_2026 = collect(df, "2026", (df["yr"] == "2022").to_numpy())
    ev = []
    for y in ["2023", "2024", "2025"]:
        ev += collect(df, y, (df["yr"].astype(int) < int(y)).to_numpy())

    tot = len([r for r in sel_2022 + sel_2026 + ev if not r["won"]])
    print(f"本命負け {tot} 件のオッズ解決（キャッシュ{len(cache)}件）…")
    for grp in (sel_2022, sel_2026, ev):
        resolve_odds(grp, cache)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))

    # グリッド全体の形（天井チェイスか・的中の薄さ）
    print("\n=== グリッド形状（Tを上げると何が起きるか）===")
    print(f"{'T':>6}{'選択ROI(22/26)':>15}{'OOF ROI(23-25)':>16}{'OOF的中数':>10}")
    sel = sel_2022 + sel_2026
    for T in GRID:
        sr, _ = roi(sel, T, "exacta")
        er, eh = roi(ev, T, "exacta")
        print(f"{T:>6}{sr*100:>14.1f}%{er*100:>15.1f}%{eh:>10}")

    # 1) 閾値
    T_all, roi_all = best_T(sel_2022 + sel_2026)
    T_22, _ = best_T(sel_2022)
    T_26, _ = best_T(sel_2026)
    print("\n=== ① 出力1: 閾値 ===")
    print(f"  {{2022,2026}}で選んだ凍結閾値 T* = {T_all}（そこでの2連単ROI {roi_all*100:.1f}%）")
    print(f"  2022単独の最適 T = {T_22} / 2026単独の最適 T = {T_26}  "
          f"→ {'一致ぎみ=頑健' if abs(T_22-T_26)<=0.3 else 'バラつき=閾値は薄い母数の産物の疑い'}")

    # 2) 23-25に凍結適用（真OOF）
    e_oof, e_hit = roi(ev, T_all, "exacta")
    t_oof, _ = roi(ev, T_all, "tansho")
    c_oof, _ = roi(ev, T_all, "combined")
    e_nofilt, _ = roi(ev, 1.0, "exacta")
    e_insample, ins_hit = roi(ev, 1.5, "exacta")
    print("\n=== ② 出力2: 23-25に凍結適用（真OOF）===")
    print(f"  凍結T*={T_all} 適用 → 2連単ROI {e_oof*100:.1f}%（的中{e_hit}） / 単勝{t_oof*100:.1f}% / 合計{c_oof*100:.1f}%")

    # 3) 差分
    print("\n=== ③ 出力3: 差分（2連単ROI）===")
    print(f"  188.7(無フィルタOOF, T=1.0)         : 実測 {e_nofilt*100:.1f}%")
    print(f"  208.3(in-sample, T=1.5を23-25で選択): 実測 {e_insample*100:.1f}%")
    print(f"  凍結OOF(T*={T_all}を{{22,26}}で選択)    : {e_oof*100:.1f}%")
    print(f"  → in-sample(1.5)からの落ち幅 : {(e_insample-e_oof)*100:+.1f}pt")
    print(f"  → 無フィルタ(1.0)からの上乗せ: {(e_oof-e_nofilt)*100:+.1f}pt")
    print("\n※Tに対しROI単調増加=sweet spot不在=実閾値なし=1.5はin-sample。")
    print("※ROI上乗せが機械効果(高配当帯集中)か割安拾い(実エッジ)かは本切り方で分離不能→②で判定。")
    print("※①はライブ保持も戦略全体の{22,26}過学習も答えない。")


if __name__ == "__main__":
    main()
