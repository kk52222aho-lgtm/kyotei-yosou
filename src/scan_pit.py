"""走査のラグ特徴を「**前日まで**」で作り直す(point-in-time で正直な版)。

## なんでこれが要るか(2026-09-27)

`test_serve_scan` で測ったら、学習時の値と**締切時点で配れる値**の一致は **73.78%**
しか無い。原因はバグやのうて情報差で、正体は **38.2% の艇が同じキー群で同じ日に
前走を持っとる**こと(1節中に1日2走が普通)。公式の K ファイルは日ごと・レース後に
出るんで、**同じ日の前のレースの着順・実STは締切時点でDBに無い**。

このまま積んだら、今日1日かけて測ったばっかりの train-serving skew を自分で作る。
打ち手は「受け入れる」やのうて**学習側も同じ形に揃える**こと。同じ日を窓から外せば
**学習と本番で計算が一致する**(定義上ズレゼロ)。代わりに情報は少し減る。

残る問いは1つだけ: **その形でも +0.27pt が残るか。**

## `scan_all.lagged` との1点のちがい

    lagged     : 窓は [max(群頭, i-w), i)          ← i = その行
    lagged_pit : 窓は [max(群頭, e-w), e)          ← e = **その日の塊の頭**

`e` は「キー群の中で、その行と同じ日付が始まる位置」。こうすると同じ日の行が
1本も窓に入らん。他は cumsum も MIN_OBS も `scan_all` と同じにしてある。

🚨 `scan_all.py` は触らん。触ったら保存済みの `scan_all.csv` /
`scan_holdout.csv`(選抜の証人)と比べられんくなる → [[feedback_incomparable_measurement]]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scan_all import LAGS, MIN_OBS, WINDOWS


def lagged_pit(df: pd.DataFrame, keycols: list[str], qcol: str):
    """(名前, 値配列) を yield。値は「**その日より前**」だけを使う。"""
    sortcols = [df["lane"].to_numpy(), df["rno"].to_numpy(), df["date"].to_numpy()]
    keyarrs = [pd.factorize(df[c])[0] for c in keycols]
    order = np.lexsort(sortcols + keyarrs[::-1])

    v = df[qcol].to_numpy(dtype=float)[order]
    ok = ~np.isnan(v)
    v0 = np.where(ok, v, 0.0)
    n = len(v)
    CS = np.concatenate([[0.0], np.cumsum(v0)])
    CM = np.concatenate([[0.0], np.cumsum(ok.astype(float))])

    kk = np.column_stack([a[order] for a in keyarrs])
    newg = np.ones(n, dtype=bool)
    newg[1:] = (kk[1:] != kk[:-1]).any(axis=1)
    gid = np.cumsum(newg) - 1
    gstart = np.flatnonzero(newg)[gid]
    i = np.arange(n)

    # 🚨 ここが唯一のちがい: **その日の塊の頭** e を窓の右端にする。
    #    群が変わった所、または日付が変わった所が「塊の頭」や。
    dts = df["date"].to_numpy()[order]
    newday = newg.copy()
    newday[1:] |= dts[1:] != dts[:-1]
    e = np.flatnonzero(newday)[np.cumsum(newday) - 1]

    inv = np.empty(n, dtype=np.int64)
    inv[order] = i

    for w in WINDOWS:
        lo = gstart if w == 0 else np.maximum(gstart, e - w)
        s = CS[e] - CS[lo]
        c = CM[e] - CM[lo]
        out = np.where(c >= MIN_OBS, s / np.where(c == 0, 1, c), np.nan)
        yield (("wall" if w == 0 else f"w{w}"), out[inv])

    # 純粋なラグも **その日の塊の頭** から数える(i-L やのうて e-L)
    for L in LAGS:
        j = e - L
        good = j >= gstart
        jj = np.where(good, j, 0)
        out = np.where(good & ok[jj], v[jj], np.nan)
        yield (f"lag{L}", out[inv])

    # 前走からの経過日数。**同じ日の前走は数えん**ので「前の日付」との差になる
    dn = df["dnum"].to_numpy()[order].astype(float)
    prev = np.full(n, np.nan)
    prev[1:] = dn[:-1]
    pe = np.where(e > gstart, prev[e], np.nan)      # その日の塊の1つ前の日
    out = np.where(e > gstart, dn - pe, np.nan)
    yield ("休み日数", out[inv])
