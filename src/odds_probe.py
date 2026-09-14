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
      1艇だけ単勝1点。順位で機械的に買わず"価格が割に合う瞬間だけ"拾う=+EVを作る唯一の手段
      (歴史では全艇のその時のオッズが無く構築不能=前向き収集でしか作れない)。
   R4(戦略トーナメント発=旅人割引): 1号がB1級&当地2連率<=8(よそ者)&全国勝率>=5(実力)の単勝1点。
      3ラウンド209戦略の探索で、Bonferroni多重比較補正(K=209)を唯一くぐった筋(ROI111%・3年独立
      108-115%・N1900・699選手)。機構=格下ラベル割引×よそ者割引の二重人間バイアスをイン残りで回収、
      地元人気=過大人気の裏面も一致。実効K>209で厳密確定は前向きのみ=これで採点。
 判定: 前向き実settleで各ルール N>=200。合格=racer_idクラスタboot生涯ROI CI下限≥100。
   kill=N>=100でCI上限<100 or N>=200で生涯ROI<90%。継続=CIが100跨ぎ。閾値を動かす=新規登録でN振り直し。
 ※スリッページ注記: 記録するオッズは"見えた瞬間"の値。実約定はその後=採点時はEVを保守側に割引く
   (張る側の指摘)。この収集器は賭けを執行しない=痕跡の記録のみ。
 ※初日(2026-07-16)の実測知見: 早い窓(T-15/T-10)はオッズ未形成の暫定値=偽の高EVが出る
   (例: T-18で3号24.5倍→T-1で7.1倍に収束、暫定EV7.08は蜃気楼)。∴全窓を録って"いつ安定
   するか"を実証する設計で正しいが、採点時は早い窓のEVを信用しない=late窓(T-3/T-1)基準で
   評価する。この発火窓の変更(all→late)を確定する時は新規事前登録でN振り直し(閾値凍結の掟)。
 ※T+3窓の追加(2026-07-23・収集の追加であって採点閾値の変更ではない): mins_to_deadline=-3
   (締切後3分)で全艇の確定オッズ O_final を録る。oddstf は締切後=確定オッズを返す(scraper参照)。
   目的は維持率 y=O_final/O_t-1 の教師ラベル(payoutsは勝艇のみ=生存バイアスで学習不能)。
   用途はオッズ維持率モデル→将来のR3'(維持率補正EV版)の"新規"事前登録のみ。既存R1-R4の採点は
   odds_probe_score の LATE=[1,3] 完全一致・export_odds の有効Nは窓{10,1}完全一致なので、
   -3行を足しても凍結済みルールには一切影響しない(締切後は賭けられない=発火対象外)。
"""
from __future__ import annotations

import datetime as dt

from . import storage, scraper, predict
from .scan import venue_name

WINDOWS = [15, 10, 5, 3, 1, -3]  # 締切の何分前の窓を狙うか。負=締切後(T+3=確定オッズO_final)
PRE_WINDOWS = [1, 3, 5, 10, 15]  # 確定窓の行にモデル確率等を引き継ぐ際の探索順(締切に近い順)
TOL = 1                          # ±分の許容


def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_timeseries (
            date TEXT, jcd TEXT, rno INTEGER,
            captured_at TEXT, mins_to_deadline INTEGER,
            bet_type TEXT, combo TEXT,       -- 'tansho'/'exacta', combo='1' or '1-3'
            odds REAL, model_p REAL, ev REAL,
            honmei INTEGER, gyaku INTEGER,   -- 本命艇, 逆イン(本命≠1号)フラグ
            reg TEXT,                        -- その艇の選手登録番号(採点のracer_idクラスタCI用)
            racer_class TEXT, nat_win REAL,  -- 級別/全国勝率(トーナメント発R4用)
            loc_2rate REAL                   -- 当地2連対率(R4=旅人割引: loc低い=よそ者の前向き検証用)
        )
    """)
    for col, typ in [("reg", "TEXT"), ("racer_class", "TEXT"), ("nat_win", "REAL"),
                     ("loc_2rate", "REAL")]:
        try:                                 # 既存テーブルへの後付けマイグレーション(冪等)
            conn.execute(f"ALTER TABLE odds_timeseries ADD COLUMN {col} {typ}")
        except Exception:
            pass
    conn.execute("""CREATE INDEX IF NOT EXISTS ix_ots ON odds_timeseries(date,jcd,rno,mins_to_deadline)""")
    # 直前情報の生テキスト置き場(タスク: 過去分は取得不能=前向きで欠測させないことが最優先)。
    # 構造化は後段。パース失敗しても raw_text 全文が必ず残る設計。
    # lap_raw/straight_raw(周回/直線展示タイム)・comment_raw(選手/記者コメント)は boatrace.jp 本体に
    # 存在しない(場公式サイト側)ためカラムだけ先行確保。場別収集器が入り次第ここに埋める。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS beforeinfo_raw (
            date TEXT, jcd TEXT, rno INTEGER,
            fetched_at TEXT, mins_to_deadline INTEGER,
            lanes_raw TEXT,      -- 艇ごと1行×6(部品交換/調整重量/前走成績ぜんぶ入り)
            parts_raw TEXT,      -- 部品交換の列挙 "2:キャブ | 5:ピストン2"
            chosei_raw TEXT,     -- 調整重量 "1:0.0 2:0.0 ..."
            st_tenji_raw TEXT,   -- スタート展示(進入隊形=並び順+ST)全文
            lap_raw TEXT,        -- 周回展示タイム(将来: 場公式サイト)
            straight_raw TEXT,   -- 直線展示タイム(将来: 場公式サイト)
            comment_raw TEXT,    -- 選手/記者コメント(将来: 場公式サイト)
            raw_text TEXT,       -- ページ本文全文(保険)
            PRIMARY KEY (date, jcd, rno)
        )
    """)
    conn.commit()


def capture_beforeinfo(conn, date, jcd, rno, mtd, stamp):
    """直前情報の生テキストを1レース1回録る。ST展示がまだ空なら後の窓で録り直す。
    失敗してもオッズ収集は殺さない(呼び元でtry)。returns 書いたか bool。"""
    row = conn.execute("SELECT st_tenji_raw FROM beforeinfo_raw WHERE date=? AND jcd=? AND rno=?",
                       (date, str(jcd), rno)).fetchone()
    if row is not None and row[0]:          # 取得済み&ST展示も入ってる=完成形
        return False
    raw = scraper.fetch_beforeinfo_raw(date, jcd, rno)
    if raw is None:                          # ページ自体が取れない時だけ諦める(次窓で再挑戦)
        return False
    conn.execute("""INSERT OR REPLACE INTO beforeinfo_raw
                    (date,jcd,rno,fetched_at,mins_to_deadline,
                     lanes_raw,parts_raw,chosei_raw,st_tenji_raw,
                     lap_raw,straight_raw,comment_raw,raw_text)
                    VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?)""",
                 (date, str(jcd), rno, stamp, mtd,
                  raw["lanes_raw"], raw["parts_raw"], raw["chosei_raw"],
                  raw["st_tenji_raw"], raw["raw_text"]))
    return True


def capture_final(conn, date, jcd, rno, mtd, stamp):
    """締切後の確定オッズ窓(mtd<0): オッズだけ実フェッチし全艇の O_final を録る。
    racelist再フェッチ/再推論はしない(展示は締切前に確定済み=直前窓の行からメタを引き継ぐ)。
    直前窓が1行も無いレース(収集器が締切前に落ちてた等)はメタNULLで録る=ラベルだけでも残す。
    returns 記録行数。"""
    wo = scraper.fetch_odds(date, jcd, rno) or {}
    if not wo:
        return 0
    prev = {}
    for w in PRE_WINDOWS:
        r = conn.execute("SELECT combo,model_p,honmei,gyaku,reg,racer_class,nat_win,loc_2rate "
                         "FROM odds_timeseries WHERE date=? AND jcd=? AND rno=? "
                         "AND mins_to_deadline=? AND bet_type='tansho'",
                         (date, str(jcd), rno, w)).fetchall()
        if r:
            prev = {int(x[0]): x for x in r}
            break
    written = 0
    for lane, od in wo.items():
        pr = prev.get(lane)
        p = pr[1] if pr else None
        conn.execute("INSERT INTO odds_timeseries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (date, str(jcd), rno, stamp, mtd, "tansho", str(lane), od, p,
                      round(p * od, 3) if p is not None else None,
                      pr[2] if pr else None, pr[3] if pr else None, pr[4] if pr else None,
                      pr[5] if pr else None, pr[6] if pr else None, pr[7] if pr else None))
        written += 1
    return written


def _mins_to(deadline_hhmm, now):
    """締切(HH:MM)までの分。過ぎてたら負。"""
    try:
        h, m = map(int, deadline_hhmm.split(":"))
    except Exception:
        return None
    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return round((dl - now).total_seconds() / 60)


def fetch_schedule(date):
    """{(jcd, rno): 'HH:MM'} 開催全レースの締切表。ループ側が30分キャッシュして使い回す用。"""
    sched = {}
    for jcd in scraper.fetch_held_venues(date):
        for rno, dl in (scraper.fetch_deadlines(date, jcd) or {}).items():
            sched[(str(jcd), rno)] = dl
    return sched


def snapshot(date, now, bundle=None, schedule=None):
    """今この瞬間、締切窓に入っているレースのオッズ+展示込みモデル確率を1回録る。
    定期(毎分)に呼ぶ想定。returns 記録件数。now/schedule は呼び出し側が渡す(再フェッチ削減)。"""
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
        if hit < 0:                          # 締切後=確定オッズ窓。beforeinfo/racelist/推論はしない
            written += capture_final(conn, date, jcd, rno, hit, stamp)
            continue
        try:                                 # 直前情報raw(タスク1)。失敗してもオッズ収集は続行
            capture_beforeinfo(conn, date, jcd, rno, hit, stamp)
        except Exception:
            pass
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
            conn.execute("INSERT INTO odds_timeseries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (date, str(jcd), rno, stamp, hit, "tansho", str(r["lane"]),
                          od, p, round(p * od, 3), int(top["lane"]), gyaku, str(r.get("reg")),
                          r.get("racer_class"),
                          float(r["nat_win"]) if r.get("nat_win") is not None else None,
                          float(r["loc_2rate"]) if r.get("loc_2rate") is not None else None))
            written += 1
    conn.commit()
    conn.close()
    return written


if __name__ == "__main__":
    n = snapshot(dt.date.today().strftime("%Y%m%d"), dt.datetime.now())
    print(f"snapshot: {n} 行記録")
