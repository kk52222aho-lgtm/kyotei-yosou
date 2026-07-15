"""勝てる形の【素の前進検証】（self-deception を排した本番テスト）。

このプロジェクトの戦略「本命≠1号艇のとき本命を単勝＋2連単上位3点」は
strategy_search.py の総当たり優勝者＝回収率を見てから選んだもの。よって
2022/2026（選択に使った区間）での高ROIは選択バイアス込み＝自己採点。

本物の検証は「戦略が一度も触っていない年」に【凍結した戦略】を当てること:
  - 戦略は固定（フィルタ=本命≠1号艇、買い目=単勝1点＋2連単上位3点。再最適化しない）
  - 学習はテスト年より前の年のみ（未来漏洩なし）
  - 数値欠損は補完せず NaN のまま（中央値リークなし。HistGBM がネイティブ処理）
  - 判定はクリーン年（=戦略選択に未使用: 2023/2024/2025）だけで下す

例:
  python -m src.validate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator

# 戦略選択(analyze.py / strategy_search.py)に使った区間。ここでの結果は自己採点。
SELECTION_YEARS = {"2022", "2026"}

# 凍結した戦略（再最適化しない）
CONTRARIAN = lambda r: r["top1"] != 1
EXACTA_N = 3


def load() -> pd.DataFrame:
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.exacta_combo, p.exacta_yen, p.tansho_lane, p.tansho_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def fit_predict_leakfree(df, tr_mask, te_mask):
    """補完リークなしで学習→予測（数値欠損は NaN のまま HGB に渡す）。"""
    X = build_frame(df, impute=False)[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    m.fit(X[tr_mask], y[tr_mask])
    out = df[te_mask].copy()
    out["p"] = m.predict_proba(X[te_mask])[:, 1]
    return out


def eval_year(test: pd.DataFrame):
    """凍結戦略で 単勝/2連単3点 の回収率を返す。"""
    tan_b = tan_h = tan_r = 0
    ex_b = ex_h = ex_r = 0
    for _, g in test.groupby(["date", "jcd", "rno"]):
        if len(g) != 6:                    # 6艇そろいのみ=飛び艇survivorship除去(監査指摘)
            continue
        g = g.sort_values("p", ascending=False)
        s = g["p"].sum()
        if s <= 0:
            continue
        p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
        top1 = int(g.iloc[0]["lane"])
        if top1 == 1:                      # 本命=1号艇は見送り（凍結戦略）
            continue
        row = g.iloc[0]
        # 単勝（本命1点）
        if not pd.isna(row["tansho_lane"]):
            tan_b += 1
            if int(row["tansho_lane"]) == top1:
                tan_h += 1
                tan_r += row["tansho_yen"]
        # 2連単 確率上位3点
        if not pd.isna(row["exacta_yen"]):
            ex = []
            for i in p:
                di = 1 - p[i]
                if di > 0:
                    for j in p:
                        if j != i:
                            ex.append((f"{i}-{j}", p[i] * p[j] / di))
            ex.sort(key=lambda x: -x[1])
            picks = [c for c, _ in ex[:EXACTA_N]]
            ex_b += len(picks)
            if row["exacta_combo"] in picks:
                ex_h += 1
                ex_r += row["exacta_yen"]
    return {
        "races": tan_b,
        "tansho_roi": tan_r / (tan_b * 100) if tan_b else float("nan"),
        "exacta_roi": ex_r / (ex_b * 100) if ex_b else float("nan"),
    }


def main():
    df = load()
    years = sorted(df["yr"].unique())
    counts = {y: int(df[df.yr == y].groupby(["date", "jcd", "rno"]).ngroups) for y in years}
    print("年別レース数:", counts)
    print("戦略選択に使った年(=自己採点):", sorted(SELECTION_YEARS))
    print("\n=== 素の前進検証（学習=テスト年より前のみ / 戦略=凍結 / 補完リークなし）===")
    print(f"{'テスト年':<8}{'区分':<10}{'賭レース':>8}{'単勝':>8}{'2連単3点':>10}")

    clean = []
    for y in years:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()   # ★テスト年より前のみ
        kind = "自己採点" if y in SELECTION_YEARS else "クリーン"
        if tr.sum() < 3000 or te.sum() < 2000:
            print(f"{y:<8}{kind:<10} 学習データ不足（前の年が無い/少ない）")
            continue
        test = fit_predict_leakfree(df, tr, te)
        r = eval_year(test)
        tag = "" if y in SELECTION_YEARS else "  ←クリーン年"
        print(f"{y:<8}{kind:<10}{r['races']:>8}{r['tansho_roi']*100:>7.0f}%"
              f"{r['exacta_roi']*100:>9.0f}%{tag}")
        if y not in SELECTION_YEARS:
            clean.append(r)

    print("\n--- 判定 ---")
    if not clean:
        print("クリーン年（2023/2024/2025）が未収集 or 学習不足。")
        print("→ 現状『検証済み』は名乗れない。収集完了後に再実行すること。")
    else:
        mt = np.nanmean([r["tansho_roi"] for r in clean])
        worst = np.nanmin([r["tansho_roi"] for r in clean])
        print(f"クリーン年 {len(clean)}年: 単勝平均{mt*100:.0f}% / 最悪{worst*100:.0f}%")
        if worst > 1.0:
            print("✓ 戦略が触っていない全クリーン年で100%超 → ここで初めて『前進検証クリア』")
        else:
            print("× クリーン年で100%割れ → 選択バイアスだった可能性。戦略は不成立")


if __name__ == "__main__":
    main()
