"""着順展開の段階別指数 γ/λ2/λ3 を実績から当てはめ直す（predict.py の定数の出どころ）。

素のHarvilleは「1着の確率ベクトルが2着・3着もそのまま支配する」と仮定する。
実測ではそれが成立せず、2着・3着ほど結果はバラける。ここでは

    P(1着=i)   ∝ p_i^γ
    P(2着=j|i) ∝ p_j^λ2   (i を除いた残りで正規化)
    P(3着=k|i,j) ∝ p_k^λ3 (i,j を除いた残りで正規化)

の3指数を最尤で当てはめ、ブートストラップで信頼区間を出す。
γ=λ2=λ3=1.0 が素のHarville。CIが1.0を外せば素のHarvilleは棄却。

【必ずモデル学習日より後の期間で回すこと】学習に使ったレースを混ぜると
確率が良く見えすぎて指数が1.0側へ引っ張られる。model.joblib の更新日を見て --start を決める。

実行例: python -m src.fit_stage_exponents --start 20260716 --end 20260806
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3

from . import predict, storage

COLS = ("lane, reg, racer_class, name, age, weight, nat_win, nat_2rate, loc_win, "
        "loc_2rate, motor_2rate, boat_2rate, tenji_time, wind_speed, wave_height")
GRID = [x / 20 for x in range(8, 61)]   # 0.40 - 3.00


def collect(start: str, end: str) -> list[tuple[dict, list[int]]]:
    """期間内の各レースについて (正規化P(1着)の辞書, 実際の1-2-3着) を返す。"""
    bundle = predict.load_model()
    if bundle is None:
        raise SystemExit("モデルが無い")
    con = sqlite3.connect(f"file:{storage.DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    races = con.execute(
        "SELECT DISTINCT e.date,e.jcd,e.rno FROM entries e JOIN payouts p "
        "ON p.date=e.date AND p.jcd=e.jcd AND p.rno=e.rno "
        "WHERE e.date>=? AND e.date<=? AND p.trifecta_combo IS NOT NULL "
        "ORDER BY e.date,e.jcd,e.rno", (start, end)).fetchall()
    out = []
    for r in races:
        date, jcd, rno = r["date"], r["jcd"], r["rno"]
        ents = [dict(x) for x in con.execute(
            f"SELECT {COLS} FROM entries WHERE date=? AND jcd=? AND rno=? ORDER BY lane",
            (date, jcd, rno)).fetchall()]
        if len(ents) != 6:
            continue
        for e in ents:
            e["jcd"] = jcd
            # 毎朝のスキャンと情報量を揃える（展示/風/波は朝は未公開）
            e["tenji_time"] = e["wind_speed"] = e["wave_height"] = None
        try:
            pr = predict.predict_entries(ents, bundle)
        except Exception:
            continue
        pay = con.execute("SELECT trifecta_combo FROM payouts WHERE date=? AND jcd=? AND rno=?",
                          (date, jcd, rno)).fetchone()
        try:
            order = [int(x) for x in pay["trifecta_combo"].split("-")]
        except (AttributeError, ValueError):
            continue
        p = {e["lane"]: e["win_prob"] for e in pr}
        s = sum(p.values()) or 1
        out.append(({k: v / s for k, v in p.items()}, order))
    return out


def _ll_win(data, g):
    return sum(math.log(max(predict.calibrated_win(p, g)[o[0]], 1e-12)) for p, o in data)


def _ll_stage(data, g, lam, stage):
    total = 0.0
    for p, o in data:
        q = predict.calibrated_win(p, g)
        taken, target = ((o[0],), o[1]) if stage == 2 else ((o[0], o[1]), o[2])
        total += math.log(max(predict._stage(q, taken, lam)[target], 1e-12))
    return total


def fit(data) -> tuple[float, float, float]:
    g = max(GRID, key=lambda x: _ll_win(data, x))
    return (g,
            max(GRID, key=lambda x: _ll_stage(data, g, x, 2)),
            max(GRID, key=lambda x: _ll_stage(data, g, x, 3)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="モデル学習日より後にすること")
    ap.add_argument("--end", required=True)
    ap.add_argument("--boot", type=int, default=200, help="ブートストラップ回数(0で省略)")
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()

    data = collect(a.start, a.end)
    print(f"対象 {len(data):,} レース ({a.start}-{a.end})", flush=True)
    if len(data) < 200:
        print("※レースが少なすぎる。区間を広げんと指数は当てにならん。")
    g, l2, l3 = fit(data)
    print(f"当てはめ: γ={g:.2f}  λ2={l2:.2f}  λ3={l3:.2f}   (素のHarvilleは全部1.00)")
    print(f"現行の定数: γ={predict.STAGE_GAMMA}  λ2={predict.STAGE_LAMBDA2}  λ3={predict.STAGE_LAMBDA3}")

    if a.boot > 0:
        random.seed(a.seed)
        res = [fit([data[random.randrange(len(data))] for _ in range(len(data))])
               for _ in range(a.boot)]
        for i, nm in enumerate(("γ (1着の鋭さ)", "λ2(2着の減衰)", "λ3(3着の減衰)")):
            v = sorted(r[i] for r in res)
            lo, hi = v[int(.025 * a.boot)], v[int(.975 * a.boot)]
            mark = "1.00を含む(素のHarvilleを棄却できん)" if lo <= 1.0 <= hi else "1.00を外す(棄却)"
            print(f"  {nm}: 中央{v[a.boot // 2]:.2f}  95%CI [{lo:.2f}, {hi:.2f}]  {mark}")


if __name__ == "__main__":
    main()
