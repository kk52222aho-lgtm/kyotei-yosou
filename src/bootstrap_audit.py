"""2連単208%の監査＋ブートストラップ（内部・スクレイプ0）。

問い（ブートが答える）: 208%は2023-25で安定した推定か、数発のファットテール的中で盛れた見かけか。
  → 3,662レースの払戻P&Lをレース単位でリサンプル（復元抽出）してROI分布とCIを出す。
問い（監査が答える）: 見かけエッジにleakage/結合ミスが無いか。
  A) 組選択(top3)は結果を参照してへんか
  B) 払戻結合は同一レースか
  C) 締切オッズと確定払戻を取り違えてへんか
  D) +19.6pt(2連単) vs +9.2pt(単勝) の偏り＝中オッズ帯集中か。帯別ROIで見る。

例:
  python -m src.bootstrap_audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_search import exacta_probs
from .validate import load, fit_predict_leakfree, SELECTION_YEARS


def build_pl(df):
    """1レース1行のP&L。tansho_return/exacta_return等。オッズ不要。"""
    recs = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            if int(top["lane"]) == 1 or pd.isna(top["exacta_yen"]):
                continue
            s = g["p"].sum()
            if s <= 0:
                continue
            p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
            top3 = [c for c, _ in exacta_probs(p)[:3]]
            e_hit = top["exacta_combo"] in top3
            recs.append({
                "date": date, "jcd": jcd, "rno": int(rno),
                "honmei": int(top["lane"]),
                "e_win": e_hit,
                "e_ret": int(top["exacta_yen"]) if e_hit else 0,
                "e_stake": 300,
                "e_combo": top["exacta_combo"],
                "e_payout": int(top["exacta_yen"]),   # 結果組の払戻（帯分けの参考）
            })
    return pd.DataFrame(recs)


def audit(df_pl, df_raw):
    print("=== 監査 ===")
    # A) 組選択の独立性: top3 は exacta_probs(モデルp) のみ。結果(e_combo)は選択後に照合。
    print("A) 組選択: top3 = exacta_probs(モデル勝率p) で決定 → 結果e_comboは top3 決定後に in 判定。"
          " 選択に結果を一切使ってへん（コード上 exacta_probs の引数は p のみ）。")
    # B) 払戻結合: entries JOIN payouts ON date,jcd,rno（validate.load）。等結合なので同一レース。
    #    重複キーが無いか確認
    dup = df_pl.duplicated(subset=["date", "jcd", "rno"]).sum()
    print(f"B) 払戻結合: date/jcd/rno 等結合。P&L内の重複キー {dup} 件（0なら1レース1行OK）。")
    # C) 締切/確定: バックテストの2連単収益は exacta_yen(=確定結果の払戻)のみ使用。
    #    事前オッズ(odds2tf)はフィルタ検証で使っただけで、このP&Lには不使用。
    print("C) 締切/確定: このP&Lの2連単収益は exacta_yen（K/結果ページの確定払戻）のみ。事前オッズ不使用＝取り違え無し。")


def band_roi(df_pl):
    print("\nD) 実結果組の払戻帯別 2連単ROI（中オッズ帯集中か）:")
    bands = [(0, 1000), (1000, 3000), (3000, 10000), (10000, 1e9)]
    labels = ["~10倍", "10-30倍", "30-100倍", "100倍~"]
    for (lo, hi), lb in zip(bands, labels):
        sub = df_pl[(df_pl.e_payout >= lo) & (df_pl.e_payout < hi)]
        if len(sub) == 0:
            continue
        roi = sub.e_ret.sum() / (len(sub) * 300)
        print(f"  結果{lb:<8} レース{len(sub):>5} 的中{sub.e_win.mean()*100:4.1f}% 回収{roi*100:6.1f}%")


def bootstrap(df_pl, B=3000, seed=0):
    rng = np.random.default_rng(seed)
    ret = df_pl.e_ret.to_numpy()
    n = len(ret)
    base = ret.sum() / (n * 300)
    rois = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        rois[b] = ret[idx].sum() / (n * 300)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    print(f"\n=== 2連単ブートストラップ（{n}レース・{B}回リサンプル）===")
    print(f"  実測ROI {base*100:.1f}% / ブート平均 {rois.mean()*100:.1f}% / 中央値 {np.percentile(rois,50)*100:.1f}%")
    print(f"  95%CI [{lo*100:.1f}% , {hi*100:.1f}%]")
    print(f"  100%割れ確率 {(rois<1.0).mean()*100:.1f}%")

    # ファットテール依存: 上位的中を除くとROIどう動くか
    hits = np.sort(df_pl.loc[df_pl.e_win, "e_ret"].to_numpy())[::-1]
    total = hits.sum()
    print("\n  的中の集中度（上位的中が総払戻に占める割合）:")
    for k in [1, 5, 10, 20]:
        if k <= len(hits):
            print(f"    上位{k:>2}的中 = 総払戻の {hits[:k].sum()/total*100:4.1f}%  "
                  f"（上位{k}除外時ROI {(total-hits[:k].sum())/(n*300)*100:5.1f}%）")
    print(f"  的中総数 {len(hits)} / 最大払戻 {int(hits[0]):,}円")


def main():
    df = load()
    pl = build_pl(df)
    raw = None
    print(f"P&L生成: {len(pl)}レース（本命≠1号艇・2連単払戻あり）\n")
    audit(pl, raw)
    band_roi(pl)
    bootstrap(pl)
    print("\n※ブート=2023-25内部の安定性。ライブ保持は別問題(papertrade)。")


if __name__ == "__main__":
    main()
