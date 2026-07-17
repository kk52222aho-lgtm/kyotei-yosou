"""前向きオッズ時系列probe: 締切前の直前オッズ×展示込みモデル確率を貯めるだけの純・収集器。

単壇の最終オッズは市場効率(K=191断面監査で確定・test_cross_section_search)。残る唯一の仮説は
「締切前の一瞬のミスプライス」で、歴史DBに時系列が無い=前向き収集しか道が無い。賭けはせず
タイムスタンプ付きでオッズ+モデル確率(締切前=展示込み)を録るだけ。Nが溜まってから事前登録
ルールで採点する(=集めてから探すと多重比較の庭になるので下記ルールを先に凍結)。

【事前登録(FROZEN・集める前に凍結。以後この定義を再最適化しない)】
 収集: 各レース締切の T-15/-10/-5/-3/-1 分の窓で、単勝(全艇)と2連単(上位)の生オッズ+その時点の
   展示込みモデル確率を1行ずつ記録(captured_at, mins_to_deadline付き)。
 採点対象の張りルール(2つだけ・事前登録):
   R1(大衆過剰反応): 本命=1号 かつ モデルP(1号)≥0.55 かつ 単勝EV=P×オッズ≥1.05 になった最初の窓で
      単勝1点。1号が展示ミス等で不当に嫌われた瞬間だけを拾う仮説。
   R2(逆イン妙味の時系列版): 本命≠1号(モデルtop) かつ 単勝EV≥1.10 の最初の窓で単勝1点。
   R3(純EVフィルタ=控除超える目を"値から構築"): 艇を問わず late窓でEV=P×オッズが最大&≥1.15の
      1艇だけ単勝1点。順位で機械的に買わず"価格that割に合う瞬間だけ"拾う=+EVを作る唯一の手段
      (歴史では全艇のその時のオッズが無く構築不能=前向き収集でしか作れない)。
 判定: 前向き実settleで各ルール N>=200。合格=racer_idクラスタboot生涯ROI CI下限≥100。
   kill=N>=100でCI上限<100 or N>=200で生涯ROI<90%。継続=CIが100跨ぎ。閾値を動かす=新規登録でN振り直し。
 ※スリッページ注記: 記録するオッズは"見えた瞬間"の値。実約定はその後=採点時はEVを保守側に割引く
   (張る側の指摘)。この収集器は賭けを執行しない=痕跡の記録のみ。
 ※初日(2026-07-16)の実測知見: 早い窓(T-15/T-10)はオッズ未形成の暫定値=偽の高EVが出る
   (例: T-18で3号24.5倍→T-1で7.1倍に収束、暫定EV7.08は蜃気楼)。∴全窓を録って"いつ安定
   するか"を実証する設計で正しいが、採点時は早い窓のEVを信用しない=late窓(T-3/T-1)基準で
   評価する。この発火窓の変更(all→late)を確定する時は新規事前登録でN振り直し(閾値凍結の掟)。
"""
from __future__ import annotations

import datetime as dt

from . import storage, scraper, predict
from .scan import venue_name

WINDOWS = [15, 10, 5, 3, 1]     # 締切の何分前の窓を狙うか
TOL = 1                          # ±分の許容


def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_timeseries (
            date TEXT, jcd TEXT, rno INTEGER,
            captured_at TEXT, mins_to_deadline INTEGER,
            bet_type TEXT, combo TEXT,       -- 'tansho'/'exacta', combo='1' or '1-3'
            odds REAL, model_p REAL, ev REAL,
            honmei INTEGER, gyaku INTEGER,   -- 本命艇, 逆イン(本命≠1号)フラグ
            reg TEXT                         -- その艇の選手登録番号(採点のracer_idクラスタCI用)
        )
    """)
    try:                                     # 既存テーブルへの後付けマイグレーション(冪等)
        conn.execute("ALTER TABLE odds_timeseries ADD COLUMN reg TEXT")
    except Exception:
        pass
    conn.execute("""CREATE INDEX IF NOT EXISTS ix_ots ON odds_timeseries(date,jcd,rno,mins_to_deadline)""")
    conn.commit()


def _mins_to(deadline_hhmm, now):
    """締切(HH:MM)までの分。過ぎてたら負。"""
    try:
        h, m = map(int, deadline_hhmm.split(":"))
    except Exception:
        return None
    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return round((dl - now).total_seconds() / 60)


def fetch_schedule(date):
    """{(jcd, rno): 'HH:MM'} 開催全レースの締切表。ループ側that30分キャッシュして使い回す用。"""
    sched = {}
    for jcd in scraper.fetch_held_venues(date):
        for rno, dl in (scraper.fetch_deadlines(date, jcd) or {}).items():
            sched[(str(jcd), rno)] = dl
    return sched


def snapshot(date, now, bundle=None, schedule=None):
    """今この瞬間、締切窓に入っているレースのオッズ+展示込みモデル確率を1回録る。
    定期(毎分)に呼ぶ想定。returns 記録件数。now/schedule は呼び出し側that渡す(再フェッチ削減)。"""
    bundle = bundle or predict.load_model()
    if schedule is None:
        schedule = fetch_schedule(date)
    conn = storage.connect()
    ensure_schema(conn)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    written = 0
    for (jcd, rno), dl in schedule.items():
        mtd = _mins_to(dl, now)
        if mtd is None:
            continue
        hit = next((w for w in WINDOWS if abs(mtd - w) <= TOL), None)
        if hit is None:
            continue
        # 既にこの窓を録ってたら二度録りしない
        got = conn.execute("SELECT 1 FROM odds_timeseries WHERE date=? AND jcd=? AND rno=? "
                           "AND mins_to_deadline=? LIMIT 1", (date, str(jcd), rno, hit)).fetchone()
        if got:
            continue
        entries = scraper.fetch_racelist(date, jcd, rno)   # 締切前=展示込み
        if not entries:
            continue
        for e in entries:
            e["jcd"] = jcd
        rows = predict.predict_entries(entries, bundle)     # 展示込みモデル確率
        top = rows[0]
        gyaku = int(top["lane"] != 1)
        wo = scraper.fetch_odds(date, jcd, rno) or {}
        for r in rows:                                       # 単勝を全艇録る(1号含む=R1/R2用)
            od = wo.get(r["lane"])
            if not od:
                continue
            p = r["win_prob"]
            conn.execute("INSERT INTO odds_timeseries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (date, str(jcd), rno, stamp, hit, "tansho", str(r["lane"]),
                          od, p, round(p * od, 3), int(top["lane"]), gyaku, str(r.get("reg"))))
            written += 1
    conn.commit()
    conn.close()
    return written


if __name__ == "__main__":
    n = snapshot(dt.date.today().strftime("%Y%m%d"), dt.datetime.now())
    print(f"snapshot: {n} 行記録")
