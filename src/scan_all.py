"""競艇 全変数 × 全キー × 全ラグ の総当たり走査。

=== なぜ今までやってへんかったか ===
このプロジェクトは **仮説を1本ずつ立てて殺す** やり方だけでやってきた。
展示CV・還元マップ・EVフィルタ・断面探索(test_cross_section_search)…全部
「先に筋を思いついて、それを検定した」もんや。**総当たりは一度も回してへん。**

行列で見たら空白が広い。既存14特徴が使う「過去」は、公式が計算済みの
累積率(nat_win/loc_win/motor_2rate)と今節(test_feature_search で7本だけ試した)だけ。
一度も着順にぶつけてへん軸がある:
  - 選手 × コース の直近実測(公式の当地率はコース別やない)
  - モーター/ボートの直近実測(公式の2連率は節をまたいだ集計で粗い)
  - 場 × コース の直近(水面は季節・改修で動く。venue_code は固定値扱い)
  - 前走からの経過日数(休み明け)
  - 純粋なラグ(1走前だけ・2走前だけ)。窓の平均で潰しとった

=== 数字を見る前に決めた設計(2026-09-02) ===

**(1) 全部レース内の順位に落とす。**
  水準どうしを比べたら場・水面・季節の共通成分が相関を作る(angou の scan.py で
  踏んだ「両方増えとる」問題のレース版)。**同じレースの6艇の中での順位**にすれば
  レース固有の交絡が消える。統計は最初から順位。

**(2) 6艇そろい & 全艇が観測できるレースだけ。**
  欠測を 0 や中央値で埋めたら「測っとらん」が「平均的」に化ける。
  特徴ごとに**使えたレース数を必ず出す**。1艇でも欠けたらそのレースは、
  その特徴については捨てる。

**(3) 測るのは2つ。生の予測力と、既存モデルへの増分。**
  a) 生: レース内順位 vs 実着(1着かどうか)
  b) 増分: レース内順位 vs **既存モデルの残差**(win - 正規化p、walk-forward OOF)
  **判定は b でやる。** a だけ見たら nat_win の焼き直しが全部生き残る。

**(4) 多重検定は max 統計で FWER。**
  8キー × 9量 × 11ラグ ≒ 800検定。p<0.05 は偶然でも40本出る。
  **日付クラスタの multiplier bootstrap** で全検定の max|t| 分布を作り、その95%点を閾値にする。
  日付でクラスタするのは、同じ日・同じ場のレースが独立やないから。
  「p<0.05 が何本」は報告せん。**閾値超えの本数だけ**。

**(5) 探索であって確認やない。**
  生き残りは仮説の候補や。結論にするには別工程(モデルに載せて、
  1号艇固定と同じ土俵で並べ直す)が要る。

usage:
  python -u -m src.scan_all panel   盤の中身・欠測を確認
  python -u -m src.scan_all base    OOF基準モデルを作って data/oof_base.csv に保存
  python -u -m src.scan_all scan    走査
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import storage

RNG = np.random.default_rng(20260902)
NBOOT = 5000
MIN_OBS = 3          # 窓平均を作るのに要る最低観測数
MIN_RACES = 3000     # これ未満しか使えん特徴は検定に出さん(検出力不足)

# ---- 走査の格子(数字を見る前に固定) --------------------------------------
WINDOWS = [0, 3, 5, 10, 20, 50]     # 0 = 全過去(expanding)
LAGS = [1, 2, 3, 5]                 # 純粋なラグ(その1走だけ)

KEYS = {
    "選手": ["reg"],
    "選手x場": ["reg", "jcd"],
    "選手xコース": ["reg", "lane"],
    "モーター": ["jcd", "motor_no"],
    "ボート": ["jcd", "boat_no"],
    "場xコース": ["jcd", "lane"],
    "選手x今節": ["reg", "jcd", "meet_id"],
    "モーターx今節": ["jcd", "motor_no", "meet_id"],
}

# 量 = 過去のレース行で観測できるもん
#
# ★2026-09-02 追記(格子を広げた理由・数字を見る前に): 走査を組む途中で
#   **実ST と 進入コースが盤に1行も入っとらん**ことが判明した(backfill_st.py 参照)。
#   競艇で一番見られとる量が欠けたまま「総当たり」を名乗るわけにいかん。
#   結果を見て足したんやのうて、**データの在庫を数えて足した**。
#   FWER は広げた後の格子全体(K)で1回だけ取る。
QTY = ["win", "top2", "top3", "fin", "bad", "tt_z", "tt_rank", "natwin", "wind",
       "st", "st_rel", "stf", "course", "maegumi", "in_gain"]


def load() -> pd.DataFrame:
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.reg, e.finish,
               e.nat_win, e.tenji_time, e.wind_speed, e.wave_height,
               e.racer_class, e.age, e.weight, e.nat_2rate,
               e.loc_win, e.loc_2rate, e.motor_2rate, e.boat_2rate
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    mb = pd.read_csv("data/motor_boat.csv",
                     dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    df["jcd"] = df["jcd"].astype(str).str.zfill(2)
    mb["jcd"] = mb["jcd"].str.zfill(2)
    df = df.merge(mb, on=["date", "jcd", "rno", "lane"], how="left")

    conn = storage.connect()
    st = pd.read_sql_query("SELECT date, jcd, rno, lane, course, st, st_f FROM results_st",
                           conn)
    conn.close()
    st["jcd"] = st["jcd"].astype(str).str.zfill(2)
    df = df.merge(st, on=["date", "jcd", "rno", "lane"], how="left")

    # 節(meet) = 同じ場で日付が連続する塊
    dts = df[["jcd", "date"]].drop_duplicates().copy()
    dts["d"] = pd.to_datetime(dts["date"], format="%Y%m%d")
    dts = dts.sort_values(["jcd", "d"])
    dts["gap"] = dts.groupby("jcd")["d"].diff().dt.days.fillna(99)
    dts["meet_id"] = dts.groupby("jcd")["gap"].transform(lambda s: (s > 1).cumsum())
    df = df.merge(dts[["jcd", "date", "meet_id"]], on=["jcd", "date"], how="left")

    df["rid"] = df["date"] + df["jcd"] + df["rno"].astype(str).str.zfill(2)
    df["dnum"] = (pd.to_datetime(df["date"], format="%Y%m%d").astype("int64")
                  // 86400_000_000_000)

    f = df["finish"]
    df["win"] = (f == 1).astype(float).where(f.notna())
    df["top2"] = (f <= 2).astype(float).where(f.notna())
    df["top3"] = (f <= 3).astype(float).where(f.notna())
    df["bad"] = f.isna().astype(float)                       # F/L/欠場
    df["fin"] = f.where(f.between(1, 6))
    df["natwin"] = pd.to_numeric(df["nat_win"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind_speed"], errors="coerce")

    # 実ST / 進入コース(backfill_st.py で K ファイルから掘ったもん)。
    # 未突合は NaN のまま。0 で埋めたら「測っとらん」が「普通のST」に化ける。
    df["st"] = pd.to_numeric(df.get("st"), errors="coerce")
    df["stf"] = pd.to_numeric(df.get("st_f"), errors="coerce")
    df["course"] = pd.to_numeric(df.get("course"), errors="coerce")
    gs = df["st"].groupby(df["rid"])
    df["st_rel"] = df["st"] - gs.transform("mean")        # レース内の相対ST
    df["maegumi"] = (df["course"] != df["lane"]).astype(float).where(df["course"].notna())
    df["in_gain"] = df["lane"] - df["course"]             # 正=内へ動いた(前づけ成功)
    tt = pd.to_numeric(df["tenji_time"], errors="coerce")
    g = tt.groupby(df["rid"])
    df["tt_z"] = (tt - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df["tt_rank"] = g.rank(method="average")

    # 6艇そろいのレースだけ(飛び艇の survivorship 除去)
    sz = df.groupby("rid")["lane"].transform("size")
    df = df[sz == 6].copy()
    for c in ("motor_no", "boat_no"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    df["meet_id"] = df["meet_id"].fillna(-1).astype(int)
    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


# ---- ラグ特徴の生成(cumsum で全窓を一気に) --------------------------------
def lagged(df, keycols, qcol):
    """(名前, 値配列) を yield。値は「そのレースより前」だけを使う。"""
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
    gstart = np.flatnonzero(newg)[gid]          # 各行の属す群の先頭 index
    i = np.arange(n)

    inv = np.empty(n, dtype=np.int64)
    inv[order] = i                               # 元の行順へ戻す索引

    for w in WINDOWS:
        lo = gstart if w == 0 else np.maximum(gstart, i - w)
        s = CS[i] - CS[lo]
        c = CM[i] - CM[lo]
        out = np.where(c >= MIN_OBS, s / np.where(c == 0, 1, c), np.nan)
        yield (("wall" if w == 0 else f"w{w}"), out[inv])

    for L in LAGS:
        j = i - L
        good = j >= gstart
        jj = np.where(good, j, 0)
        out = np.where(good & ok[jj], v[jj], np.nan)
        yield (f"lag{L}", out[inv])

    # 前走からの経過日数(そのキーで)
    dn = df["dnum"].to_numpy()[order].astype(float)
    prev = np.full(n, np.nan)
    prev[1:] = dn[:-1]
    out = np.where(i > gstart, dn - prev, np.nan)
    yield ("休み日数", out[inv])


# ---- レース内順位 z 化 ------------------------------------------------------
def race_z(vals, nrace):
    """全6艇が観測できたレースだけ、レース内順位を中心化して返す。他は NaN。

    行は date/jcd/rno/lane 順で 6行ずつ連続しとる前提(load が保証)。
    同値は平均順位。1艇でも欠けたらそのレースは丸ごと NaN(0埋めせん)。
    """
    V = vals.reshape(nrace, 6)
    full = (~np.isnan(V)).all(axis=1)
    out = np.full((nrace, 6), np.nan)
    if full.any():
        S = V[full]
        lt = (S[:, None, :] < S[:, :, None]).sum(2)      # 自分より小さい艇の数
        eq = (S[:, None, :] == S[:, :, None]).sum(2)     # 同値(自分を含む)
        out[full] = 1.0 + lt + (eq - 1) * 0.5 - 3.5      # 平均順位を中心化(レース内の和=0)
    return out.ravel()


def build_base(df):
    """既存14特徴の walk-forward OOF 予測を保存(増分の分母)。"""
    from sklearn.calibration import CalibratedClassifierCV
    from .features import FEATURES, build_frame
    from .train import _make_estimator
    conn = storage.connect()
    raw = pd.read_sql_query("SELECT * FROM entries WHERE win IS NOT NULL", conn)
    conn.close()
    raw["jcd"] = raw["jcd"].astype(str).str.zfill(2)
    raw["yr"] = raw["date"].str[:4]
    X = build_frame(raw, impute=False)[FEATURES].to_numpy(dtype=float)
    y = raw["win"].to_numpy(dtype=int)
    out = []
    for yv in sorted(raw["yr"].unique()):
        te = (raw["yr"] == yv).to_numpy()
        tr = (raw["yr"].astype(int) < int(yv)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            print(f"[skip] {yv}")
            continue
        print(f"[fit] {yv} train={tr.sum():,} test={te.sum():,}", flush=True)
        m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        o = raw.loc[te, ["date", "jcd", "rno", "lane"]].copy()
        o["p"] = m.predict_proba(X[te])[:, 1]
        out.append(o)
    pd.concat(out).to_csv("data/oof_base.csv", index=False)
    print("saved data/oof_base.csv")


def scan(df, cutoff=None, out="data/scan_all.csv"):
    """cutoff を渡すと、**その年より前のレースだけ**で統計を取る。

    走査で特徴を選んで、同じレースで採点したら自己採点や([[feedback_in_sample_rule_scoring]])。
    cutoff='2025' で走査を 2023-2024 に閉じ、確認は 2025-2026 だけでやる。
    """
    oof = pd.read_csv("data/oof_base.csv",
                      dtype={"date": str, "jcd": str, "rno": int, "lane": int})
    oof["jcd"] = oof["jcd"].str.zfill(2)
    df = df.merge(oof, on=["date", "jcd", "rno", "lane"], how="left")
    rid_codes, uniq = pd.factorize(df["rid"], sort=False)
    nrace = len(uniq)
    # race_z は「6行ずつ連続」を前提にしとる。merge で崩れてへんことを実際に確かめる。
    assert len(df) == nrace * 6, f"行数が6の倍数やない: {len(df)} vs {nrace}*6"
    assert (rid_codes.reshape(nrace, 6) == rid_codes[::6][:, None]).all(), "レースの塊が崩れとる"

    p = df["p"].to_numpy(dtype=float)
    win = df["win"].to_numpy(dtype=float)
    have = ~np.isnan(p) & ~np.isnan(win)
    if cutoff:
        have &= (df["date"].str[:4] < cutoff).to_numpy()
        print(f"走査を {cutoff} より前に限定(採点用の年は一切見ん)")
    cnt = np.bincount(rid_codes, weights=have.astype(float), minlength=nrace)
    keep = cnt[rid_codes] == 6
    print(f"基準モデルOOFが全6艇そろうレース: {int(keep.sum() // 6):,}")

    psum = np.bincount(rid_codes, weights=np.where(keep, p, 0.0), minlength=nrace)
    q = np.where(keep, p / np.maximum(psum[rid_codes], 1e-12), np.nan)
    resid = np.where(keep, win - q, np.nan)          # 増分用の残差
    raw_r = np.where(keep, win - 1 / 6, np.nan)      # 生の予測力用

    # ---- 🚨 2026-09-02 追加: 「増分」の相手が弱すぎた問題の手当て -------------
    # 偽薬(scan_placebo.py)で判明: **既存モデルに入っとる nat_win / tenji_time が、
    # モデル残差に対して増分 t=+17.0 / -10.1 を出す。**器の壊れやのうて、
    # 既存モデル(艇ごとの二値分類+isotonic)が、自分が持っとる特徴の
    # **レース内順位の情報を使い切っとらん**いうことや。
    # このままやと「新しい変数を見つけた」やのうて「モデルが手持ちを使い切っとらん」を
    # 大量に拾う。実際 912/1320 が生き残った。
    # → 残差を、既存特徴の**レース内順位**が張る空間へ直交化してから当てる。
    #   これで判定は「線形順位で使い切った既存情報の、さらに外にあるか」になる。
    #   相手を強くする方向やから保守側や。
    BASE_RANK = ["lane", "class_num", "age", "weight",
                 "nat_win", "nat_2rate", "loc_win", "loc_2rate",
                 "motor_2rate", "boat_2rate", "tenji_time",
                 "tt_rank", "tt_gap", "inner_pow_edge"]
    from .features import build_frame
    bf = build_frame(df, impute=False)
    Z = []
    for c in BASE_RANK:
        zc = race_z(pd.to_numeric(bf[c], errors="coerce").to_numpy(float), nrace)
        Z.append(zc)
    Z.append(np.where(keep, q, np.nan))          # モデル確率そのもの(非線形分を残す)
    Z = np.column_stack(Z)
    okz = keep & ~np.isnan(Z).any(axis=1)
    okz = (np.bincount(rid_codes, weights=okz.astype(float),
                       minlength=nrace)[rid_codes] == 6)
    print(f"既存特徴の順位が全艇そろうレース: {int(okz.sum() // 6):,} "
          f"(採点対象の {okz.sum() / max(keep.sum(), 1):.1%})")
    Zk = Z[okz]
    beta, *_ = np.linalg.lstsq(Zk, resid[okz], rcond=None)
    resid_o = np.full(len(df), np.nan)
    resid_o[okz] = resid[okz] - Zk @ beta
    r2 = 1 - np.var(resid_o[okz]) / np.var(resid[okz])
    print(f"既存特徴の順位が残差から説明した割合: {r2:.2%}  ← ここが0でない分だけ"
          f"「モデルが手持ちを使い切っとらん」")

    dcodes, dun = pd.factorize(df["dnum"], sort=True)
    nd = len(dun)
    print(f"日クラスタ数: {nd:,}")

    names, W_inc, W_raw, nraces = [], [], [], []
    W_ort, nrort = [], []
    for kname, kcols in KEYS.items():
        for q_ in QTY:
            for lname, vals in lagged(df, kcols, q_):
                z = race_z(vals, nrace)
                m = keep & ~np.isnan(z)
                nrz = int(m.sum() // 6)
                if nrz < MIN_RACES:
                    continue
                zi = np.where(m, z, 0.0)
                names.append(f"{kname}|{q_}|{lname}")
                nraces.append(nrz)
                W_inc.append(np.bincount(dcodes, weights=zi * np.where(m, resid, 0.0),
                                         minlength=nd))
                W_raw.append(np.bincount(dcodes, weights=zi * np.where(m, raw_r, 0.0),
                                         minlength=nd))
                mo = m & okz
                W_ort.append(np.bincount(dcodes, weights=np.where(mo, z, 0.0)
                                         * np.where(mo, resid_o, 0.0), minlength=nd))
                nrort.append(int(mo.sum() // 6))
        print(f"  [{kname}] 累計 {len(names)} 特徴", flush=True)

    W_inc = np.array(W_inc).T            # (nd, K)
    W_raw = np.array(W_raw).T
    W_ort = np.array(W_ort).T
    K = W_inc.shape[1]
    print(f"\n検定対象 K={K}(欠測/検出力で落ちたもんを除く)")

    def tstats(W):
        den = np.sqrt((W ** 2).sum(0))
        return W.sum(0) / np.where(den == 0, np.nan, den)

    t_inc, t_raw, t_ort = tstats(W_inc), tstats(W_raw), tstats(W_ort)

    def fwer(W, nboot=NBOOT):
        den = np.sqrt((W ** 2).sum(0))
        den = np.where(den == 0, np.nan, den)
        mx = np.empty(nboot)
        step = 500
        for a in range(0, nboot, step):
            b = min(step, nboot - a)
            xi = RNG.standard_normal((b, W.shape[0]))
            mx[a:a + b] = np.nanmax(np.abs((xi @ W) / den), axis=1)
        return float(np.percentile(mx, 95))

    thr_inc = fwer(W_inc)
    thr_raw = fwer(W_raw)
    thr_ort = fwer(W_ort)
    print(f"FWER 5% 閾値(max|t| の95%点): 生 {thr_raw:.2f} / モデル残差 {thr_inc:.2f} / "
          f"**直交化残差 {thr_ort:.2f}**")
    print(f"参考: 単独 p<0.05 は 1.96。それで数えたら偶然でも約 {0.05 * K:.0f} 本出る")

    res = pd.DataFrame({"feature": names, "t_raw": t_raw, "t_inc": t_inc,
                        "t_ort": t_ort, "races": nraces, "races_ort": nrort})
    res["survive_raw"] = np.abs(res["t_raw"]) > thr_raw
    res["survive_inc"] = np.abs(res["t_inc"]) > thr_inc
    res["survive_ort"] = np.abs(res["t_ort"]) > thr_ort
    res = res.sort_values("t_ort", key=np.abs, ascending=False)
    res.to_csv(out, index=False)

    print("\n=== 結果 ===")
    print(f"  生の予測力で閾値超え : {int(res['survive_raw'].sum())} / {K}")
    print(f"  **増分で閾値超え**   : {int(res['survive_inc'].sum())} / {K}")
    print("\n上位25(|t_inc|順):")
    print(res.head(25).to_string(index=False))
    print("\n生は強いのに増分が閾値未満(=既存モデルが既に吸収しとる)上位10:")
    ab = res[~res["survive_inc"]].sort_values("t_raw", key=np.abs, ascending=False)
    print(ab.head(10).to_string(index=False))
    print("\n保存: data/scan_all.csv")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "panel"
    df = load()
    print(f"盤: {len(df):,}行 / {df['rid'].nunique():,}レース / "
          f"{df['date'].nunique():,}日 ({df['date'].min()}-{df['date'].max()})")

    if cmd == "panel":
        print("\n量の欠測:")
        for q_ in QTY:
            print(f"  {q_:10s} 非欠測 {df[q_].notna().mean():6.1%}")
        print("\nキーの群数:")
        for name, cols in KEYS.items():
            print(f"  {name:14s} {df.groupby(cols).ngroups:,}群")
        n = len(KEYS) * len(QTY) * (len(WINDOWS) + len(LAGS) + 1)
        print(f"\n格子: {len(KEYS)}キー x {len(QTY)}量 x "
              f"{len(WINDOWS) + len(LAGS) + 1}ラグ = {n:,} 検定")
    elif cmd == "base":
        build_base(df)
    elif cmd == "holdout":
        # 走査は 2023-2024 だけ。2025-2026 は一切見ん(確認用に取っとく)
        scan(df, cutoff="2025", out="data/scan_holdout.csv")
    else:
        scan(df)


if __name__ == "__main__":
    main()
