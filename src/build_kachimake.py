"""勝負駆けフラグの遡及構築(タスク2)。

節(連続開催日クラスタ)を復元し、節内累積得点率から準優ボーダー(上位18名)との
順位差を出して「予選最終日にボーダー際にいる=勝負駆け」フラグを全歴史レースに付与する。

定義(この近似で凍結。タスク3の事前登録検証が参照する):
 - 節ID: 同一jcdの開催日を日付順に並べ、前日と連続(diff=1日)なら同節。
 - 得点率: 着順点 1着10/2着8/3着6/4着4/5着2/6着1、非完走(99)=0点。
   分母=出走数(finishがある走のみ)。事故点(F等の減点)は無視(近似)。
 - 順位: その日の【朝時点】=前日までの累積得点率で節内全選手をランク(rate降順, method='min')。
   →未来リーク無し(当日の成績は当日の順位に入らない)。
 - 予選最終日: 節長Lに対し L-2 日目(6日節=4日目)。L<4 の節はフラグ対象外。
 - kachimake = (予選最終日) かつ (順位が18位ボーダー±3位以内 = 15〜21位)。
 - border_diff = rank - 18 (負=ボーダー内側)。H4の突破安全圏= border_diff<=-5。

出力: kyotei.db の kachimake テーブル(date,jcd,rno,laneで entries と結合可)。
実行: python -u -m src.build_kachimake
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import storage

POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
BORDER = 18          # 準優ボーダー(得点率上位18名)
NEAR = 3             # ボーダー±3位以内で勝負駆け
SAFE = 5             # H4: ボーダー+5位以上=突破安全圏


def assign_setsu(dates: list[str]) -> dict[str, int]:
    """日付リスト(YYYYMMDD, 同一jcd)→ {date: 節連番}。連続日=同節。"""
    out, sid, prev = {}, 0, None
    for d in sorted(dates):
        cur = dt.date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        if prev is None or (cur - prev).days != 1:
            sid += 1
        out[d] = sid
        prev = cur
    return out


def main():
    conn = storage.connect()
    df = pd.read_sql_query(
        "SELECT date, jcd, rno, lane, reg, finish FROM entries "
        "WHERE reg IS NOT NULL AND reg != ''", conn)
    print(f"entries {len(df):,}行 {df['date'].min()}-{df['date'].max()}")

    # 1. 節ID
    df["setsu"] = ""
    for jcd, g in df.groupby("jcd"):
        m = assign_setsu(g["date"].unique().tolist())
        df.loc[g.index, "setsu"] = g["date"].map(lambda d: f"{jcd}-{m[d]}")

    # 節内の日付インデックスと節長
    day_idx = {}
    setsu_len = {}
    for setsu, g in df.groupby("setsu"):
        days = sorted(g["date"].unique())
        setsu_len[setsu] = len(days)
        for i, d in enumerate(days, 1):
            day_idx[(setsu, d)] = i
    df["day"] = [day_idx[(s, d)] for s, d in zip(df["setsu"], df["date"])]
    df["slen"] = df["setsu"].map(setsu_len)
    print(f"節数 {df['setsu'].nunique():,} / 節長分布 "
          f"{df.groupby('setsu')['slen'].first().value_counts().sort_index().to_dict()}")

    # 2-3. 前日までの累積得点率→朝時点ランク→ボーダー差 (day>=2 の日に付与)
    df["pts"] = df["finish"].map(POINTS).fillna(0.0)
    df["started"] = df["finish"].notna().astype(int)

    out = []
    for setsu, g in df.groupby("setsu"):
        L = int(g["slen"].iloc[0])
        if L < 3:
            continue
        qual_final = L - 2 if L >= 4 else None
        # 選手×日の集計 → 日次累積
        pr = (g[g["started"] == 1].groupby(["reg", "day"])
              .agg(pts=("pts", "sum"), n=("started", "sum")).reset_index())
        # 各日dの朝時点: day<d の累積
        for d in sorted(g["day"].unique()):
            if d < 2:
                continue
            hist = pr[pr["day"] < d].groupby("reg").agg(
                pts=("pts", "sum"), n=("n", "sum"))
            hist = hist[hist["n"] > 0]
            if len(hist) < BORDER:      # 選手数がボーダー未満の変則節はスキップ
                continue
            hist["rate"] = hist["pts"] / hist["n"]
            hist["rank"] = hist["rate"].rank(ascending=False, method="min").astype(int)
            ranks = hist["rank"].to_dict()
            rates = hist["rate"].to_dict()
            ns = hist["n"].to_dict()
            day_rows = g[g["day"] == d]
            for _, r in day_rows.iterrows():
                rk = ranks.get(r["reg"])
                if rk is None:
                    continue
                bd = rk - BORDER
                isq = int(qual_final is not None and d == qual_final)
                out.append((r["date"], r["jcd"], int(r["rno"]), int(r["lane"]),
                            r["reg"], setsu, d, L, int(ns[r["reg"]]),
                            round(rates[r["reg"]], 3), rk, bd, isq,
                            int(isq and abs(bd) <= NEAR)))

    res = pd.DataFrame(out, columns=[
        "date", "jcd", "rno", "lane", "reg", "setsu", "day", "slen",
        "races_before", "cum_rate", "rank", "border_diff", "is_qual_final",
        "kachimake"])
    conn.execute("DROP TABLE IF EXISTS kachimake")
    res.to_sql("kachimake", conn, index=False)
    conn.execute("CREATE INDEX ix_km ON kachimake(date,jcd,rno,lane)")
    conn.commit()
    print(f"kachimake {len(res):,}行 保存 / フラグ1 {int(res['kachimake'].sum()):,}艇 "
          f"/ 予選最終日行 {int(res['is_qual_final'].sum()):,}")
    print(res[res["kachimake"] == 1].groupby(res["date"].str[:4]).size().to_dict())
    conn.close()


if __name__ == "__main__":
    main()
