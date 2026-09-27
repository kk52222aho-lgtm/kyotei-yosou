"""走査24本(pit版)を表に置く。**学習と本番が同じ値を読む**ための1か所。

## なんで表に置くか

pit版(`scan_pit.lagged_pit`)の値は「**その日より前**」だけで決まるんで、
いっぺん計算した過去の行は二度と変わらん。せやから表に持てる。

表を1つにすると、学習(`train.py`)と本番(`predict_race`)が**同じ行を読む**んで
train-serving skew が構造的に起きん。計算をそれぞれの場所に置いたら、
片方だけ直した日にズレる。今日それを1日かけて測ったばっかりや
([[insight_empty_column_blames_the_fetcher]] の隣の失敗)。

## 列名でズレを起こさんように

列は `f00`〜`f23` いう素な名前で置いて、**どの index が何の特徴か**を
`scan_spec` に書く。読む側は `spec_sha` を突き合わせて、ちがったら**止まる**。
走査を選び直したら sha が変わるんで、古いモデルに新しい列を食わせる事故が起きん。

## 今日のレースをどう入れるか

`scan_all.load()` は `JOIN payouts` やから**結果が出とるレースだけ**。今日のレースは
入らんので、`entries` から足す(量は NaN)。pit の窓はその日より前しか見んので
今日の量が無くても今日の値は正しく出る。**`meet_id` は足した後に組み直す**
(今日の日付を含めんと「今節」が別物になって履歴ゼロに化ける)。

usage:
    python -u -m src.build_scan_features            # 直近45日を書く
    python -u -m src.build_scan_features --all      # 全期間を書き直す
"""
from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from . import storage
from .scan_all import KEYS, load as scan_load
from .scan_pit import lagged_pit
from .test_scan_confirm import pick

KEPT_SRC = "data/scan_holdout.csv"
NCOL = 24
COLS = [f"f{i:02d}" for i in range(NCOL)]
# クラウドへ持っていく写しの範囲。予想に要るんは今日だけやが、日付の境目のために2日。
# 🚨 gzip + 5桁丸めにしとる。素のCSVは3日で770KBで、**毎日コミットしたら年280MB**。
#    行ごとに spec_sha(64字)を書いとったんが一番効いとった → 12字に切る
RECENT_DAYS = 2
RECENT_CSV = "data/scan_recent.csv.gz"

DDL = f"""
CREATE TABLE IF NOT EXISTS scan_spec (
  idx INTEGER, name TEXT, spec_sha TEXT, built_at TEXT,
  PRIMARY KEY (idx, spec_sha)
);
CREATE TABLE IF NOT EXISTS scan_features (
  date TEXT, jcd TEXT, rno INTEGER, lane INTEGER,
  {", ".join(c + " REAL" for c in COLS)},
  spec_sha TEXT, built_at TEXT,
  PRIMARY KEY (date, jcd, rno, lane)
);
CREATE INDEX IF NOT EXISTS ix_scan_features_date ON scan_features(date);
"""


def kept_specs() -> list[tuple[str, str, str]]:
    """載せる24本の (キー, 量, ラグ)。

    選抜は **2023-2024 に閉じて 2025-2026 で検証した方**(`scan_holdout.csv`)を使う。
    🚨 その選抜自体は「同じ日の情報が入った計算」で行われとる。pit で選び直したら
       別の24本になる可能性は残っとる(保守側やない)。
    """
    kept = pick(pd.read_csv(KEPT_SRC))
    return [(r["key"], r["qty"], r["lag"]) for _, r in kept.iterrows()]


def spec_sha(specs) -> str:
    return hashlib.sha256(
        "\n".join(f"{k}|{q}|{l}" for k, q, l in specs).encode()).hexdigest()


def _meet_id(df: pd.DataFrame) -> pd.Series:
    """場ごとに日付が連続する塊へ番号を振る(`scan_all.load` と同じ規則)。"""
    dts = df[["jcd", "date"]].drop_duplicates().copy()
    dts["d"] = pd.to_datetime(dts["date"], format="%Y%m%d")
    dts = dts.sort_values(["jcd", "d"])
    dts["gap"] = dts.groupby("jcd")["d"].diff().dt.days.fillna(99)
    dts["meet_id"] = dts.groupby("jcd")["gap"].transform(lambda s: (s > 1).cumsum())
    return df.merge(dts[["jcd", "date", "meet_id"]], on=["jcd", "date"],
                    how="left", suffixes=("_old", ""))["meet_id"]


def build_frame(specs) -> pd.DataFrame:
    """結果が出とるレース + **まだ出とらんレース**をつないだ盤を返す。"""
    df = scan_load()
    conn = storage.connect()
    ent = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, reg, racer_class, age, weight, nat_win,"
        " nat_2rate, loc_win, loc_2rate, motor_2rate, boat_2rate, tenji_time,"
        " wind_speed, wave_height FROM entries", conn)
    conn.close()
    ent["jcd"] = ent["jcd"].astype(str).str.zfill(2)
    k = ["date", "jcd", "rno", "lane"]
    have = set(map(tuple, df[k].to_numpy()))
    m = ~pd.Series(list(map(tuple, ent[k].to_numpy()))).isin(have)
    add = ent[m.to_numpy()].copy()
    if len(add):
        mb = pd.read_csv("data/motor_boat.csv",
                         dtype={"date": str, "jcd": str, "rno": int, "lane": int})
        mb["jcd"] = mb["jcd"].str.zfill(2)
        add = add.merge(mb, on=k, how="left")
        for c in ("motor_no", "boat_no"):
            add[c] = pd.to_numeric(add.get(c), errors="coerce").fillna(-1).astype(int)
        # 量は全部 NaN。結果はまだ出とらん。**0で埋めん**
        for c in df.columns:
            if c not in add.columns:
                add[c] = np.nan
        add["rid"] = add["date"] + add["jcd"] + add["rno"].astype(str).str.zfill(2)
        add["dnum"] = (pd.to_datetime(add["date"], format="%Y%m%d").astype("int64")
                       // 86400_000_000_000)
        # 6艇そろっとるレースだけ(盤の作りに合わせる)
        sz = add.groupby("rid")["lane"].transform("size")
        add = add[sz == 6]
        df = pd.concat([df, add[df.columns]], ignore_index=True)
    # 🚨 meet_id は足した後に組み直す。今日の日付を入れんと「今節」が別物になる
    df["meet_id"] = _meet_id(df).fillna(-1).astype(int)
    n_add = len(add) if len(add) else 0
    print(f"盤 {len(df):,}行(結果ありに {n_add:,}艇を足した / "
          f"{df['date'].min()}〜{df['date'].max()})")
    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45, help="書き込む日数(既定45)")
    ap.add_argument("--all", action="store_true", help="全期間を書き直す")
    a = ap.parse_args()

    specs = kept_specs()
    sha = spec_sha(specs)
    print(f"走査 {len(specs)}本 / spec_sha={sha[:16]}…")
    assert len(specs) == NCOL, f"列数が合わん: {len(specs)} != {NCOL}"

    df = build_frame(specs)
    vals = {}
    for i, (key, qty, lag) in enumerate(specs):
        got = None
        for nm, v in lagged_pit(df, KEYS[key], qty):
            if nm == lag:
                got = v
                break
        assert got is not None, f"{key}|{qty}|{lag} が出てこん"
        vals[COLS[i]] = got
    F = pd.DataFrame(vals)
    for c in ("date", "jcd", "rno", "lane"):
        F[c] = df[c].to_numpy()

    days = sorted(df["date"].unique())
    tgt = days if a.all else days[-a.days:]
    W = F[F["date"].isin(set(tgt))]
    print(f"書く範囲: {tgt[0]}〜{tgt[-1]}({len(tgt)}日) / {len(W):,}艇")

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn = storage.connect()
    conn.executescript(DDL)
    conn.executemany("INSERT OR REPLACE INTO scan_spec VALUES (?,?,?,?)",
                     [(i, f"{k}|{q}|{l}", sha, now) for i, (k, q, l) in enumerate(specs)])
    ph = ",".join("?" * (4 + NCOL + 2))
    rows = [tuple(r) + (sha, now) for r in
            W[["date", "jcd", "rno", "lane"] + COLS].itertuples(index=False, name=None)]
    # 🚨 159万行を1トランザクションで書いたら **錠を数分握る**。
    #    `odds_probe`(締切前の1分刻み探査)が `database is locked` で
    #    13:55〜15:35 に38回こけた。busy_timeout=120秒でも足りん。
    #    1分刻みの列は逃した分が戻らんので、**刻んで commit して錠を放す**。
    CH = 20_000
    for i in range(0, len(rows), CH):
        conn.executemany(f"INSERT OR REPLACE INTO scan_features VALUES ({ph})",
                         rows[i:i + CH])
        conn.commit()
        if len(rows) > CH and (i // CH) % 20 == 0:
            print(f"    書いた {min(i + CH, len(rows)):,}/{len(rows):,}", flush=True)

    q = conn.execute("SELECT COUNT(*), MIN(date), MAX(date),"
                     " COUNT(DISTINCT spec_sha) FROM scan_features").fetchone()
    print(f"scan_features {q[0]:,}艇 / {q[1]}〜{q[2]} / spec_sha の種類 {q[3]}")
    if q[3] != 1:
        print("  🚨 spec_sha が複数在る。古い版が混じっとる。--all で書き直す")
    # 非欠測(24本とも揃っとる艇)。学習時に測った 59.9% と近いはず
    got = conn.execute(
        f"SELECT AVG(CASE WHEN {' AND '.join(c + ' IS NOT NULL' for c in COLS)}"
        " THEN 1.0 ELSE 0.0 END) FROM scan_features").fetchone()[0]
    print(f"  24本とも揃っとる艇: {got:.1%}  (検定時は 59.9%)")
    # 🚨 クラウド(streamlit.app)は kyotei.db を持っとらん。表が唯一の**計算元**やが、
    #    直近だけは写しをリポに置いて予測が止まらんようにする。
    #    写しは計算をやり直さん(2つ目の出どころにはせん)。spec_sha も一緒に書いて
    #    読む側が版ちがいを見つけられるようにしとく。
    recent = sorted(df["date"].unique())[-RECENT_DAYS:]
    Rc = F[F["date"].isin(set(recent))].copy()
    Rc[COLS] = Rc[COLS].round(5)
    Rc["spec_sha"] = sha[:12]
    Rc[["date", "jcd", "rno", "lane"] + COLS + ["spec_sha"]].to_csv(
        RECENT_CSV, index=False, compression="gzip")
    import os as _os
    print(f"  写し: {RECENT_CSV}  {len(Rc):,}艇 / {recent[0]}〜{recent[-1]}"
          f" / {_os.path.getsize(RECENT_CSV):,} bytes")

    for d in sorted(df["date"].unique())[-3:]:
        n = conn.execute("SELECT COUNT(*) FROM scan_features WHERE date=?", (d,)).fetchone()[0]
        g = conn.execute(
            f"SELECT SUM(CASE WHEN {' AND '.join(c + ' IS NOT NULL' for c in COLS)}"
            " THEN 1 ELSE 0 END) FROM scan_features WHERE date=?", (d,)).fetchone()[0]
        print(f"    {d}: {n:5,}艇 / 24本そろい {g or 0:5,}")
    conn.close()


if __name__ == "__main__":
    main()
