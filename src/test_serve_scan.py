"""走査24本を**本番で配れるか**を、過去レースで学習時の値と突き合わせて測る。

## なんで先にこれをやるか

偽薬を通して走査の +0.27pt は情報やと確かめた(`test_scan_noise`)。せやが
本番に積むには `predict_race` へ値を配らなあかんくて、そこに2つ落とし穴が在る:

1. `scan_all.load()` は `JOIN payouts WHERE tansho_lane IS NOT NULL` =
   **結果が在るレースだけ**取る。今日のレースは入らんので手で足す
2. キーに `選手x今節`/`モーターx今節` が在る。`meet_id` は「同じ場で日付が連続する塊」
   やから、**今日の日付を含めて数えんと今節が別物**になる(履歴ゼロ=NaN に化ける)

そして本当の問いはこれや。`lagged` は構造上「そのレースより前」だけを使うんで、
**同じ日の前のレースの結果がDBに入っとるか**で値が変わる。締切の時点で
公式の K ファイルはまだ出とらん(K は日ごと・レース後)。つまり本番で配れるんは
**前日までの履歴**で、学習時は**同じ日の前のレースまで**入っとる。ズレる。

## 測り方(数字を見る前に決めた)

日付 D をいくつか取って、同じ D の全レースを3通りで組んで比べる:

    train  : 盤ぜんぶ(学習時にその行が受け取った値)
    serve_D: D より前の行 + D の行(量は NaN)。**同じ日の前のレースが無い**
             = 締切時点でほんまに配れる形

突き合わせるんは値そのものと、**本命が替わるか**(不一致率)。
値がちがうだけやと大きさが分からんんで、最後は必ず決定で測る
→ [[insight_lookahead_that_does_not_move_the_decision]] の裏返し。

usage: python -u -m src.test_serve_scan [日付の本数]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .scan_all import KEYS, lagged, load as scan_load
from .test_scan_confirm import pick

RNG = np.random.default_rng(20260927)
KEPT_SRC = "data/scan_holdout.csv"


def kept_specs() -> list[tuple[str, str, str]]:
    """載せる24本の (キー, 量, ラグ) を holdout 選抜から取る。

    本番に積むなら**選抜を 2023-2024 に閉じて 2025-2026 で検証した方**を使う。
    全年で選び直した `scan_all.csv` は自己採点になる。
    """
    res = pd.read_csv(KEPT_SRC)
    kept = pick(res)
    return [(r["key"], r["qty"], r["lag"]) for _, r in kept.iterrows()]


def add_scan(df: pd.DataFrame, specs) -> list[str]:
    """df に S_* 列を足して列名を返す。df は書き換える。"""
    names = []
    for key, qty, lag in specs:
        for lname, vals in lagged(df, KEYS[key], qty):
            if lname == lag:
                nm = f"S_{key}_{qty}_{lag}"
                df[nm] = vals
                names.append(nm)
                break
    assert len(names) == len(specs), "走査特徴の組み立てが合わん"
    return names


def main() -> None:
    ndate = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    specs = kept_specs()
    full = scan_load()
    print(f"\n盤 {len(full):,}行 / {full['date'].nunique():,}日  走査 {len(specs)}本")

    names = add_scan(full, specs)
    # 比べる日付は「盤の後ろの方」から。前の方やと履歴が薄うて本番と形がちがう
    days = sorted(full["date"].unique())[-200:]
    pickd = sorted(RNG.choice(days, size=min(ndate, len(days)), replace=False))
    print(f"比べる日付: {', '.join(pickd)}")

    rows = []
    for d in pickd:
        # 締切時点でほんまに配れる形: D より前 + D の行(量は NaN)
        past = full[full["date"] < d]
        today = full[full["date"] == d].copy()
        # 🚨 量を NaN にする。結果はまだ出とらん。**落とすんやのうて欠損にする**
        #    (落としたら今節の meet_id 群から今日の行が消えてキーが変わる)
        QTY = sorted({q for _, q, _ in specs})
        for q in QTY:
            today[q] = np.nan
        srv = pd.concat([past, today], ignore_index=True)
        snames = add_scan(srv, specs)
        got = srv[srv["date"] == d]
        exp = full[full["date"] == d]
        k = ["date", "jcd", "rno", "lane"]
        m = exp[k + names].merge(got[k + snames], on=k, suffixes=("_t", "_s"))
        assert len(m) == len(exp), f"{d}: 突合で行が合わん"
        for nm in names:
            a, b = m[nm + "_t"].to_numpy(float), m[nm + "_s"].to_numpy(float)
            both = ~np.isnan(a) & ~np.isnan(b)
            rows.append({
                "date": d, "feature": nm, "n": len(a),
                "同じ": int((np.isclose(a, b, equal_nan=True)).sum()),
                "片方だけ欠損": int((np.isnan(a) ^ np.isnan(b)).sum()),
                "値がちがう": int((both & ~np.isclose(a, b)).sum()),
                "最大差": float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0,
            })
    R = pd.DataFrame(rows)
    g = R.groupby("feature")[["n", "同じ", "片方だけ欠損", "値がちがう"]].sum()
    g["最大差"] = R.groupby("feature")["最大差"].max()
    g["一致率"] = g["同じ"] / g["n"]
    g = g.sort_values("一致率")
    print(f"\n=== 学習時の値 vs 締切時点で配れる値  ({len(pickd)}日 / 各行{len(names)}本) ===")
    print(f"{'特徴':34s} {'一致率':>8s} {'片方欠損':>8s} {'値ちがい':>8s} {'最大差':>10s}")
    for nm, r in g.iterrows():
        mk = "  🚨" if r["一致率"] < 0.99 else ""
        print(f"{nm:34s} {r['一致率']:8.2%} {int(r['片方だけ欠損']):8,} "
              f"{int(r['値がちがう']):8,} {r['最大差']:10.4f}{mk}")
    tot = g["n"].sum()
    print(f"\n全体: {g['同じ'].sum():,}/{tot:,} = {g['同じ'].sum()/tot:.2%} 一致 / "
          f"片方だけ欠損 {int(g['片方だけ欠損'].sum()):,} / 値がちがう {int(g['値がちがう'].sum()):,}")
    print("\n🚨 一致せん分は**バグやのうて情報のちがい**や(同じ日の前のレースの結果が"
          "締切時点では無い)。次に測るんは「本命が替わるか」で、そこまで出さんと"
          "大きさが分からん。")


if __name__ == "__main__":
    main()
