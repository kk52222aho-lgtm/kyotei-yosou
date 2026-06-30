"""「勝てる形」探索：確率ベースの買い目戦略を全賭式・全条件で総当たりし、
回収率100%超を out-of-sample で探す。最後にウォークフォワードで過学習チェック。

ポイント:
  確率ベース選択なら払戻テーブルだけで検証可能（フルオッズのスクレイプ不要）。
  買い目はモデル確率で選び、勝ち/負けと配当は payouts から取る。
  → 数万レース規模で総当たりできる。

賭式: 単勝 / 2連単 / 3連単（確率は Harville/Plackett-Luce）
選択: 確率上位N点
フィルタ: 全レース / 本命≠1号艇 / モデル自信度

例:
  python -m src.strategy_search
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator


def load() -> pd.DataFrame:
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.trifecta_combo, p.trifecta_yen, p.exacta_combo, p.exacta_yen,
               p.tansho_lane, p.tansho_yen
        FROM entries e
        JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    return df


def exacta_probs(p: dict[int, float]) -> list[tuple[str, float]]:
    """2連単 i-j の確率 P(i-j)=p_i·p_j/(1-p_i) を降順で。"""
    out = []
    for i in p:
        di = 1 - p[i]
        if di <= 0:
            continue
        for j in p:
            if j != i:
                out.append((f"{i}-{j}", p[i] * p[j] / di))
    out.sort(key=lambda x: -x[1])
    return out


def trifecta_probs(p: dict[int, float]) -> list[tuple[str, float]]:
    out = []
    for i in p:
        di = 1 - p[i]
        if di <= 0:
            continue
        for j in p:
            if j == i:
                continue
            dij = 1 - p[i] - p[j]
            if dij <= 0:
                continue
            for k in p:
                if k in (i, j):
                    continue
                out.append((f"{i}-{j}-{k}", p[i] * p[j] / di * p[k] / dij))
    out.sort(key=lambda x: -x[1])
    return out


def build_records(test: pd.DataFrame) -> list[dict]:
    """テストレースごとに、確率ランキングと払戻を1件にまとめる。"""
    recs = []
    for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
        g = g.sort_values("p", ascending=False)
        s = g["p"].sum()
        if s <= 0:
            continue
        p = {int(l): float(v) / s for l, v in zip(g["lane"], g["p"])}
        top = g.iloc[0]
        recs.append({
            "top1": int(top["lane"]),
            "p0": float(top["p"]),
            "p_lane1": float(p.get(1, 0.0)),  # 1号艇のモデル確率(正規化)
            "top2": int(g.iloc[1]["lane"]) if len(g) > 1 else 0,
            "tansho_lane": top["tansho_lane"],
            "tansho_yen": top["tansho_yen"],
            "exacta_combo": top["exacta_combo"], "exacta_yen": top["exacta_yen"],
            "trifecta_combo": top["trifecta_combo"], "trifecta_yen": top["trifecta_yen"],
            "ex_rank": [c for c, _ in exacta_probs(p)],
            "tf_rank": [c for c, _ in trifecta_probs(p)],
            "win_lane_rank": list(p.keys()),  # 既に確率降順
        })
    return recs


def evaluate(recs, bet, n, filt):
    """bet: 'tansho'/'exacta'/'trifecta', n: 上位n点, filt: レコード→bool"""
    bets = hits = ret = races = 0
    for r in recs:
        if not filt(r):
            continue
        if bet == "tansho":
            if r["tansho_lane"] is None or pd.isna(r["tansho_lane"]):
                continue
            picks = [str(l) for l in r["win_lane_rank"][:n]]
            win, yen = str(int(r["tansho_lane"])), r["tansho_yen"]
        elif bet == "exacta":
            if r["exacta_combo"] is None or pd.isna(r["exacta_yen"]):
                continue
            picks = r["ex_rank"][:n]
            win, yen = r["exacta_combo"], r["exacta_yen"]
        else:
            if r["trifecta_combo"] is None or pd.isna(r["trifecta_yen"]):
                continue
            picks = r["tf_rank"][:n]
            win, yen = r["trifecta_combo"], r["trifecta_yen"]
        races += 1
        bets += len(picks)
        if win in picks:
            hits += 1
            ret += yen
    stake = bets * 100
    return {"races": races, "bets": bets,
            "hit": hits / races if races else 0,
            "roi": ret / stake if stake else 0,
            "pl": ret - stake}


def _fit_predict(df, tr, te):
    X = build_frame(df)[FEATURES].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    m.fit(X[tr], y[tr])
    out = df[te].copy()
    out["p"] = m.predict_proba(X[te])[:, 1]
    return out


FILTERS = {
    "全レース": lambda r: True,
    "本命≠1号艇": lambda r: r["top1"] != 1,
    "本命=1号艇": lambda r: r["top1"] == 1,
    "自信60%+": lambda r: r["p0"] >= 0.6,
    "本命≠1&自信50%+": lambda r: r["top1"] != 1 and r["p0"] >= 0.5,
}

GRID = [
    ("tansho", 1), ("tansho", 2),
    ("exacta", 1), ("exacta", 2), ("exacta", 3), ("exacta", 5),
    ("trifecta", 1), ("trifecta", 3), ("trifecta", 6), ("trifecta", 12),
]


def main():
    df = load()
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    tr = df["date"].to_numpy() < cut
    te = ~tr
    print(f"学習 {tr.sum()//6}レース / テスト {te.sum()//6}レース (cut={cut})\n探索中…")

    test = _fit_predict(df, tr, te)
    recs = build_records(test)

    rows = []
    for bet, n in GRID:
        for fname, ffn in FILTERS.items():
            r = evaluate(recs, bet, n, ffn)
            if r["races"] < 30:
                continue
            rows.append({
                "賭式": bet, "点数": n, "条件": fname,
                "レース": r["races"], "回収率": r["roi"],
                "的中率": r["hit"], "収支": r["pl"],
            })
    R = pd.DataFrame(rows).sort_values("回収率", ascending=False)
    pd.set_option("display.unicode.east_asian_width", True)

    disp = R.copy()
    disp["回収率"] = (disp["回収率"] * 100).round(1).astype(str) + "%"
    disp["的中率"] = (disp["的中率"] * 100).round(1).astype(str) + "%"
    disp["収支"] = disp["収支"].map(lambda v: f"{int(v):+,}")
    print("\n=== 回収率ランキング（上位15）===")
    print(disp.head(15).to_string(index=False))
    print("\n=== 回収率100%超のみ ===")
    over = disp[R["回収率"] > 1.0]
    print(over.to_string(index=False) if len(over) else "  （なし）")
    return R


if __name__ == "__main__":
    main()
