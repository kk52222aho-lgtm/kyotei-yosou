"""トーナメント最終判定官: エージェントthaが出した全戦略を holdout(2025) で審判+多重比較補正。

生成はdevだけ見て行う。ここで初めてholdout(未使用の2025)を開き、試した総数Kで Bonferroni 補正
(CIを 1-0.05/K 水準に広げる)。生き残り = dev>100 かつ holdout Bonferroni-CI下限>100。
＝「バックテストで勝った」でなく「未来(holdout)を、試行回数を割り引いても超えた」奴だけが本物。

使い方: python -m src.strategy_judge queries.json   (queriesは文字列のJSON配列)
"""
from __future__ import annotations

import json
import sys
import numpy as np

from . import strategy_eval as se


def judge(queries):
    sub = se.load()
    uniq = list(dict.fromkeys(q for q in queries if isinstance(q, str)))
    K = max(len(uniq), 1)
    bonf = 100 * (1 - 0.05 / K)
    rows = []
    for q in uniq:
        try:
            sel = sub.query(q).copy()
        except Exception:
            continue
        sel["_ret"] = np.where(sel["won"] == 1, sel["tansho_yen"], 0.0)
        dev = sel[sel["yr"].isin(se.DEV)]
        hp = sel[sel["yr"] == se.HOLD]
        if len(hp) < 80 or len(dev) < 80:
            continue
        droi = dev["_ret"].sum() / (len(dev) * 100) * 100
        hroi = hp["_ret"].sum() / (len(hp) * 100) * 100
        lo95, hi95 = se._cluster_ci(hp, pct=95)
        loB, hiB = se._cluster_ci(hp, pct=bonf)
        rows.append({"q": q, "dev_roi": round(droi, 1), "hold_roi": round(hroi, 1),
                     "hold_n": len(hp), "ci95": [round(lo95), round(hi95)],
                     "bonf": [round(loB), round(hiB)],
                     "survive": bool(droi > 100 and loB > 100)})
    rows.sort(key=lambda r: -r["hold_roi"])
    print(f"審判: {len(rows)}戦略 (ユニークK={K}, Bonferroni={bonf:.2f}%水準)")
    print(f"多重比較: 偶然のnaive95%生存の期待≈{0.05*K:.1f}個\n")
    print("hold_ROI  dev_ROI  N     CI95        Bonferroni    生存  戦略")
    for r in rows[:25]:
        s = "★" if r["survive"] else " "
        print(f"  {r['hold_roi']:5.1f}   {r['dev_roi']:5.1f}  {r['hold_n']:5d} "
              f"[{r['ci95'][0]:3d},{r['ci95'][1]:3d}]  [{r['bonf'][0]:3d},{r['bonf'][1]:3d}] {s}  {r['q']}")
    surv = [r for r in rows if r["survive"]]
    print(f"\n=== 生存(dev>100 & Bonferroni-CI下限>100): {len(surv)}個 ===")
    if surv:
        for r in surv:
            print(f"  ★ {r['q']}  hold{r['hold_roi']}% Bonf{r['bonf']}")
        print("→ 前向きprobeへ(未来が最終審判)。")
    else:
        print("→ 全戦略が篩落ち=devで勝ったのは過学習/偶然。運営の壁は割れず(今回の探索範囲では)。")
    return rows


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "queries.json"
    with open(src, encoding="utf-8") as f:
        queries = json.load(f)
    judge(queries)


if __name__ == "__main__":
    main()
