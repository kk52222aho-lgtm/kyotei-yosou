"""戦略の決定論的審判機: フィルタ式(=どの艇を単勝で買うか)を dev/holdout で採点。

エージェントが捏造できないよう評価はここに一元化。戦略=pandas query文字列(基盤の列に対する条件)。
その条件に合う艇を単勝1点で買う。dev(2023-24)で開発、holdout(2025)で審判(選択に未使用)。
多重比較補正は judge 側(試した総数)で行う=ここは1戦略を正直に採点するだけ。

使い方: evaluate("rank==1 and conf<45")  /  CLI: python -m src.strategy_eval "rank==2 and jcd=='12'"
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

_SUB = None
DEV = ["2023", "2024"]
HOLD = "2025"


def load():
    global _SUB
    if _SUB is None:
        _SUB = pd.read_pickle("data/strategy_substrate.pkl")
    return _SUB


def _cluster_ci(g, iters=1500, pct=95):
    ret = g["_ret"].to_numpy(float)
    regs = g["reg"].to_numpy()
    uniq = pd.unique(regs)
    idx = {u: np.where(regs == u)[0] for u in uniq}
    rng = np.random.default_rng(17)
    rois = []
    for _ in range(iters):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        sel = np.concatenate([idx[u] for u in pick])
        rois.append(ret[sel].sum() / (len(sel) * 100) * 100)
    lo = (100 - pct) / 2
    return float(np.percentile(rois, lo)), float(np.percentile(rois, 100 - lo))


def evaluate(query, min_n=80, dev_only=False):
    """戦略(query)を採点。返り: dev/holdout の N・ROI・的中、holdoutのクラスタCI。
    dev_only=True は holdout を伏せる(生成中のエージェントに未来を覗かせない=過学習防止)。"""
    sub = load()
    try:
        sel = sub.query(query)
    except Exception as e:
        return {"valid": False, "err": f"query error: {e}", "query": query}
    sel = sel.copy()
    sel["_ret"] = np.where(sel["won"] == 1, sel["tansho_yen"], 0.0)

    def stats(part):
        n = len(part)
        if n == 0:
            return {"n": 0, "roi": None, "hit": None}
        return {"n": int(n), "roi": round(part["_ret"].sum() / (n * 100) * 100, 1),
                "hit": round(part["won"].mean() * 100, 1)}

    dev = stats(sel[sel["yr"].isin(DEV)])
    by_yr = {y: stats(sel[sel["yr"] == y]) for y in DEV}       # 年別=年跨ぎ安定の判定用
    dev["by_year"] = {y: s["roi"] for y, s in by_yr.items()}
    dev["stable"] = all((s["roi"] or 0) > 100 for s in by_yr.values())  # 各年>100か
    if dev_only:
        return {"valid": dev["n"] >= min_n, "query": query, "dev": dev,
                "holdout": "HIDDEN(dev生成中は審判年を伏せる)"}
    hp = sel[sel["yr"] == HOLD]
    hold = stats(hp)
    ci = _cluster_ci(hp) if hold["n"] >= min_n else (None, None)
    return {"valid": hold["n"] >= min_n and dev["n"] >= min_n, "query": query,
            "dev": dev, "holdout": {**hold, "ci_lo": None if ci[0] is None else round(ci[0], 1),
                                    "ci_hi": None if ci[1] is None else round(ci[1], 1)}}


def main():
    import json
    args = sys.argv[1:]
    dev_only = "--dev" in args
    qs = [a for a in args if a != "--dev"] or ["rank==1"]
    res = [evaluate(q, dev_only=dev_only) for q in qs]   # 複数式=1回のpickleロードで一括採点
    print(json.dumps(res if len(res) > 1 else res[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
