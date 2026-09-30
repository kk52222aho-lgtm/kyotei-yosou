"""オリジナル展示データ (一周/まわり足/直線タイム) の収集。

BOATCAST が全24場の「オリジナル展示データ」を単一ホストの静的TSVで集約している。
公式 boatrace.jp の出走表/直前情報には出ない、各場が独自計測した数値。

エンドポイント (認証なし・素のGETで200・1レース約230-265バイト):
    https://race.boatcast.jp/txt/{jcd}/bc_oriten_{YYYYMMDD}_{jcd}_{RR}.txt   (RR=2桁ゼロ埋め)

形式 (先頭 data=、ヘッダ行にラベル、以降 艇番→選手名→数値):
    data=
    1\t3
    一　周\tまわり足\t直　線
    1\t南　　　佑典\t37.44\t6.26\t7.39
    ...
列構成は場依存 (尼崎は2列 等)。ヘッダのラベルで対応付けし、想定外ラベルは raw に退避。

運営への配慮:
- まず getEncodeList2_Race_{date}.json (全場1リクエスト/日) で開催レースだけ列挙し、
  存在しないレースを叩かない (403の乱射を避ける)。
- リクエスト間に SLEEP_SEC の待機。
- 取得済み (date,jcd,rno) は oriten_done に記録し、再実行では再取得しない。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date as _date, datetime, timedelta

import requests

# 2026-09-08: 素の `import storage` は src/ を cwd にした時しか通らんかった。
# タスク登録して `python -m src.collect_oriten` で呼べるように相対importへ。
from .venues import ALL_JCD, name as venue_name
from . import storage

HOST = "https://race.boatcast.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (kyotei-yosou research; personal use)"}
# 🚨 2026-09-24: 1.5秒やと約14発で頻度制限に当たる(25発の実測で15〜21発目が403)。
# 2.0秒でもまだ引っかかる(1パスで進まんかった)。2.5秒で8/8、3.0秒に置いた。
# **速さやのうて弾かれんのが目的**や。取り戻すんは急がん——
# blocked は済み扱いにせんので、毎日の常駐が何日かかけて拾う。
SLEEP_SEC = 3.0  # マナーとしての待機 (公式より一段丁寧に)
# 403(頻度制限)を食らった時の冷却。実測で約10秒で戻るんで余裕を持たせる
BLOCK_WAIT_SEC = 15.0
# 連続でこれだけ弾かれたら、その日は畳んで出直す(粘っても向こうの迷惑なだけ)
MAX_CONSEC_BLOCKED = 3
# 403 がこの回数続いたら「不在」とみなして打ち止め(毎晩叩き続けんため)。
# **仮定やのうて規則**として書いとく: 5回叩いて5回とも403なら不在扱い
MAX_ATTEMPTS = 5
# 古い日付は2回で打ち止め(根拠は _mark のコメント: 1,553回叩いて0本)
MAX_ATTEMPTS_OLD = 2
OLD_DAYS = 30

_session = requests.Session()
_session.headers.update(HEADERS)

# ヘッダラベル (全角空白を除去して正規化) -> カラム名
# 注意: 場ごとにラベルもスケールも異なる。桐生は「一周」でなく「半周ラップ」を公開し、
# まわり足の値域も大村~6 と 住之江/尼崎~11.5 で桁が違う(計測定義が場依存)。
# したがって生値の場横断比較は不可。特徴量はレース内順位/正規化で使うこと。
LABEL_MAP = {
    "一周": "lap",
    "半周ラップ": "halflap",
    "まわり足": "mawariashi",
    "直線": "straight",
}

ORITEN_SCHEMA = """
CREATE TABLE IF NOT EXISTS oriten (
    date       TEXT NOT NULL,
    jcd        TEXT NOT NULL,
    rno        INTEGER NOT NULL,
    lane       INTEGER NOT NULL,
    lap        REAL,
    halflap    REAL,
    mawariashi REAL,
    straight   REAL,
    raw        TEXT,
    fetched_at TEXT,
    PRIMARY KEY (date, jcd, rno, lane)
);
"""

DONE_SCHEMA = """
CREATE TABLE IF NOT EXISTS oriten_done (
    date   TEXT NOT NULL,
    jcd    TEXT NOT NULL,
    rno    INTEGER NOT NULL,
    -- 'ok'    … 中身が取れた
    -- 'empty' … 200 やのに中身が無い(ほんまに不在)
    -- 'blocked' … 403。**不在か弾かれか、コードだけでは区別でけへん**(下)
    -- 'gone'  … 403 が MAX_ATTEMPTS 回続いた。不在とみなして打ち止め
    -- 'cancelled' … **公式Kが「走っとらん」と言うとる**。通信ゼロで打ち止め。
    --            gone(諦めた)と分けとくんは、後から見分けられるようにするため
    -- 'error' … それ以外
    status TEXT,
    n_lane INTEGER,
    attempts INTEGER DEFAULT 0,
    fetched_at TEXT,
    PRIMARY KEY (date, jcd, rno)
);
"""


def _init(conn) -> None:
    # ロック競合時は最大60秒待つ(他プロセスのDB利用と共存)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(ORITEN_SCHEMA)
    conn.execute(DONE_SCHEMA)
    # 既存DBに halflap 列が無ければ追加
    cols = {r[1] for r in conn.execute("PRAGMA table_info(oriten)")}
    if "halflap" not in cols:
        conn.execute("ALTER TABLE oriten ADD COLUMN halflap REAL")
    dcols = {r[1] for r in conn.execute("PRAGMA table_info(oriten_done)")}
    if "attempts" not in dcols:
        conn.execute("ALTER TABLE oriten_done ADD COLUMN attempts INTEGER DEFAULT 0")
    conn.commit()


def _get_text(url: str, retries: int = 2) -> tuple[int, str]:
    """(status_code, text) を返す。最終試行の status を返す。"""
    status = 0
    for i in range(retries):
        try:
            r = _session.get(url, timeout=20)
            status = r.status_code
            if status == 200:
                return 200, r.content.decode("utf-8", "replace")
            if status == 403:
                # 🚨 2026-09-24: ここで「403 = 不在。リトライ不要」と決め打っとった。
                #    実測では **403 は頻度制限**で、不在やない。
                #    1.5秒間隔で叩くと **14発で引っかかって、約10秒で戻る**
                #    (25発の実測: 200×18 / 403×7、15発目〜21発目が403)。
                #    empty と記帳された 9/17 以降のレースを叩き直したら
                #    **5/5 が 200 で中身を返した**。8日ぶん静かに落としとった。
                #    冷却を待って同じレースをやり直す。それでも駄目なら blocked。
                time.sleep(BLOCK_WAIT_SEC * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(SLEEP_SEC * (i + 1))
    return status, ""


def enumerate_day(date: str) -> dict[str, int]:
    """getEncodeList2 で当日開催の {jcd: 最大レース番号} を返す (全場1リクエスト)。"""
    url = f"{HOST}/api_txt/getEncodeList2_Race_{date}.json"
    status, text = _get_text(url)
    out: dict[str, int] = {}
    if status != 200 or not text:
        return out
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return out
    for r in d.get("return_info", []):
        jcd = str(r.get("RaceStudiumNo", "")).zfill(2)
        hi = r.get("Holding_info", [])
        if hi:
            n = len(hi[0].get("Status_Info", []))
            if n:
                out[jcd] = n
    return out


def parse_oriten(text: str) -> list[dict]:
    """oriten TSV をパースして [{lane, lap, mawariashi, straight, raw}] を返す。"""
    lines = [ln for ln in text.split("\n")]
    # data= を除去
    lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("data=")]
    # ヘッダ行を探す (一/まわり/直 を含む)
    header_idx = None
    labels: list[str] = []
    for i, ln in enumerate(lines):
        cells = ln.split("\t")
        norm = [c.replace("　", "").replace(" ", "") for c in cells]
        if any(k in "".join(norm) for k in ("一周", "まわり足", "直線")):
            header_idx = i
            labels = norm
            break
    if header_idx is None:
        return []
    rows = []
    for ln in lines[header_idx + 1:]:
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        try:
            lane = int(parts[0])
        except ValueError:
            continue
        if lane < 1 or lane > 6:
            continue
        values = parts[2:]  # 選手名の後が数値列
        rec = {"lane": lane, "lap": None, "halflap": None,
               "mawariashi": None, "straight": None, "raw": ln}
        for label, val in zip(labels, values):
            col = LABEL_MAP.get(label)
            v = None
            try:
                v = float(val)
            except (ValueError, TypeError):
                v = None
            if col:
                rec[col] = v
        rows.append(rec)
    return rows


def _control_ok(conn) -> bool:
    """台帳で ok と分かっとるレースを1本叩いて、いま弾かれとるかを見る。

    🚨 403 は「不在」と「弾かれ」の両方に使われとる(2026-09-24 実測)。
    **中身を持っとると分かっとる玉**を対照にすれば、その場で分けられる。
    これが無いと、ほんまに不在の場日で永久に畳み続ける。
    """
    row = conn.execute(
        "SELECT date, jcd, rno FROM oriten_done WHERE status='ok' "
        "ORDER BY date DESC LIMIT 1").fetchone()
    if not row:
        return False
    d, j, r = row
    time.sleep(SLEEP_SEC)
    st, _ = _get_text(f"{HOST}/txt/{j}/bc_oriten_{d}_{j}_{str(r).zfill(2)}.txt")
    return st == 200


def fetch_race(conn, date: str, jcd: str, rno: int) -> str:
    """1レース分を取得してDBへ。status ('ok'|'empty'|'error') を返す。"""
    rr = str(rno).zfill(2)
    url = f"{HOST}/txt/{jcd}/bc_oriten_{date}_{jcd}_{rr}.txt"
    status, text = _get_text(url)
    now = datetime.now().isoformat(timespec="seconds")
    if status == 200 and text:
        rows = parse_oriten(text)
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO oriten "
                "(date,jcd,rno,lane,lap,halflap,mawariashi,straight,raw,fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(date, jcd, rno, r["lane"], r["lap"], r["halflap"],
                  r["mawariashi"], r["straight"], r["raw"], now) for r in rows],
            )
            _mark_done(conn, date, jcd, rno, "ok", len(rows), now)
            return "ok"
        _mark_done(conn, date, jcd, rno, "empty", 0, now)
        return "empty"
    if status == 200:
        # 200 やのに中身が無い = ほんまに不在。ここだけが empty
        _mark_done(conn, date, jcd, rno, "empty", 0, now)
        return "empty"
    # 🚨 403/その他は「不在」やない。**blocked として残して次回やり直す**
    _mark_done(conn, date, jcd, rno, "blocked" if status == 403 else "error", 0, now)
    return "blocked" if status == 403 else "error"


def _mark_done(conn, date, jcd, rno, status, n_lane, now) -> None:
    """🚨 403 は「不在」と「弾かれ」の両方に使われとる。コードだけでは分けられん。

    分ける唯一の方法は**日を変えて何回か叩くこと**や。実測(2026-09-24):

      9月の empty … 叩き直したら 5/5 が 200(=弾かれとっただけ)
      4月の empty … 3秒間隔でも 6/6 が 403 のまま。しかもその日その場は
                    ok が1本も無い(=ほんまに不在)

    せやから 403 は `blocked` で残して次回やり直すが、**MAX_ATTEMPTS 回続いたら
    `gone`(不在とみなす)に落として打ち止めにする。**
    こうせんと不在のレースを毎晩叩き続けることになる。
    """
    prev = conn.execute(
        "SELECT attempts FROM oriten_done WHERE date=? AND jcd=? AND rno=?",
        (date, jcd, rno)).fetchone()
    att = (prev[0] or 0) if prev else 0
    if status == "blocked":
        att += 1
        # 🚨 2026-09-30: **古い日付は2回で打ち止め。**5回いう規則は「403が頻度制限
        #    かもしれん」から来とったが、いまは対照(_control_ok)が同じ秒に200を
        #    返して判別しとる。そして実測が在る——夜の回収の run 5回ぶんで
        #    **叩いた 1,553 / 取れた 0**(同じ期間の窓の run は 1,170叩いて 1,587本取れとる
        #    ので、器やのうて古い分が向こうに無いだけ)。
        #    残り約915本を5回ずつやと約4,500リクエストで期待収穫は0。
        #    直近は据え置く(公開が遅れる筋と頻度制限が実際に在る)。
        try:
            age = (_date.today() - datetime.strptime(date, "%Y%m%d").date()).days
        except ValueError:
            age = 0
        limit = MAX_ATTEMPTS_OLD if age > OLD_DAYS else MAX_ATTEMPTS
        if att >= limit:
            status = "gone"
    else:
        att = 0
    conn.execute(
        "INSERT OR REPLACE INTO oriten_done "
        "(date,jcd,rno,status,n_lane,attempts,fetched_at) VALUES (?,?,?,?,?,?,?)",
        (date, jcd, rno, status, n_lane, att, now),
    )


def _already_done(conn, date, jcd, rno) -> bool:
    """済みは ok と empty(=200やのに中身が無い)だけ。

    blocked/error は**済みやない**。弾かれただけのもんを「不在」に格上げしたら、
    そのレースは永久に取れんくなる。→ [[insight_zero_is_a_measurement]]
    """
    r = conn.execute(
        "SELECT 1 FROM oriten_done WHERE date=? AND jcd=? AND rno=? "
        "AND status IN ('ok','empty','gone','cancelled')",
        (date, jcd, rno)).fetchone()
    return r is not None


# 🚨 2026-09-29: **出しとらん場**。江戸川(場03)は 4月〜9月の83日・951レースで
#    ok が**1本もゼロ**やった。同じ日・同じ秒に場02を叩いたら 200 で中身も在る
#    (対照つきの実弾1発で確かめた)。**弾かれとるんやのうて出しとらん。**
#    未回収 2,129 のうち 951(45%)がこの1場で、毎晩そこを叩き続けとった。
#
#    決め打ちの禁止リストにはせん。**台帳から数える**(試した数が十分で ok がゼロ)。
#    ただし**完全に目を塞がん**——1回につき1本だけ様子見に行く。出し始めたら気付ける。
#    → [[insight_zero_is_a_measurement]] / [[feedback_missing_param_looks_structural]]
FIRST_DATA_HOUR = 11     # これより前に今日の分を聞いても在り得ん
SILENT_MIN_TRIED = 60      # これだけ試して
SILENT_PROBE = 1           # ok が0なら飛ばす。ただし毎回1本だけ様子見
SKIP_JCD: set[str] = set()


def silent_venues(conn, min_tried: int = SILENT_MIN_TRIED) -> dict[str, tuple[int, int]]:
    """{場: (試した数, ok数)} で ok が1本も無い場を返す。"""
    out = {}
    for j, n, ok in conn.execute(
            "SELECT jcd, COUNT(*), SUM(status='ok') FROM oriten_done GROUP BY jcd"):
        if n >= min_tried and not (ok or 0):
            out[j] = (n, ok or 0)
    return out


def mark_cancelled_gone(conn, verbose: bool = True) -> int:
    """**中止レースを通信ゼロで打ち止めにする。**

    🚨 2026-09-30: 滞留939本を公式Kと突き合わせたら **84本は中止**やった。
       中止のレースに展示データが在るわけないのに、**5回叩いてから諦めとった**
       (84×5 = 420リクエストの無駄)。**手元に在る権威ある出どころで分かるもんを
       他人に聞くな。**

    見分け方(記憶にある印そのまま): 中止は**その日の後半に連続して出る**。
    実例 場03 20260428 → R1-R7 は着順6/6、**R8-R12 は着順0/6**。

    条件:
      ① **その日のどこかの場に着順が在る**(= K ファイルが入っとる証拠)
      ② そのレース自身の着順が **1本も無い**
      ③ 日付が2日以上前(当日の途中を「中止」と読まんため)

    🚨 ①を「同じ場日」やのうて「その日」に広げる時、**手元のKが欠けとるだけの日**を
       中止と読む危険が在った(記憶: 正常サイズやのに中身が欠けとる日が4日、
       09-16 は欠け23.9%)。**一番怪しい例で実際に確かめた**:
       20260603 は場03・07・09・10 の**4場まとめてゼロ**で、天候にも取りこぼしにも見える。
       K を退避して取り直したら **同じ結果**で、しかも **バイト単位で同一**やった
       = サーバ側も同じ中身 → 取りこぼしやのうて**ほんまに中止**。
       → [[insight_estimator_needs_ground_truth]]

    `gone`(5回叩いて諦めた)と `cancelled`(公式が走っとらんと言うとる)は
    **別の状態に分けとる**。打ち止めは取り消せんので、後から理由が見分けられるように。
    """
    rows = conn.execute("""
        SELECT d.date, d.jcd, d.rno FROM oriten_done d
        JOIN (SELECT date, jcd, rno, SUM(finish IS NOT NULL) fin
              FROM entries GROUP BY date, jcd, rno) e
          ON e.date = d.date AND e.jcd = d.jcd AND e.rno = d.rno
        JOIN (SELECT date, SUM(finish IS NOT NULL) datefin
              FROM entries GROUP BY date) v
          ON v.date = d.date
        WHERE d.status NOT IN ('ok', 'empty', 'gone', 'cancelled')
          AND e.fin = 0 AND v.datefin > 0
          AND d.date < ?
    """, ((_date.today() - timedelta(days=2)).strftime("%Y%m%d"),)).fetchall()
    if not rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "UPDATE oriten_done SET status='cancelled', fetched_at=? "
        "WHERE date=? AND jcd=? AND rno=?",
        [(now, d, j, r) for d, j, r in rows])
    conn.commit()
    if verbose:
        print(f"  中止レースを通信ゼロで打ち止め: {len(rows):,}本"
              f"(公式Kで着順が1本も無い=展示が在るわけない)", flush=True)
    return len(rows)


def collect_day(conn, date: str, jcds: list[str] | None = None,
                verbose: bool = True) -> dict[str, int]:
    """1日分を収集。{'ok','empty','error','skip'} の件数を返す。"""
    # やることが無い日は**通信せず**飛ばす。
    # 🚨 403 の退きを長くしたら、列挙の1発にも 15+30+45+60 秒かかるようになって、
    #    済みだけの日を舐めるのに何時間もかかった。まず台帳で用事の有無を見る。
    # 🚨 2026-09-30: **まだ走っとらんレースの展示データは在るわけない。**
    #    毎朝07:30の run が今日の約150本に403を連打して attempts を無駄に積んどった。
    #    実測(9/18〜9/29): 同日中に取れるんは 144本中 **5〜21本**(0本の日も多い)で、
    #    同日の初取得は **08:58〜10:32**。残り約130本は**翌朝に取れとる**。
    #    → 午前は今日を聞かん。午後に回した時は同日分も取りに行く余地を残す。
    #    (未来日は常に飛ばす。存在し得んもんを叩いて attempts を積むと
    #     MAX_ATTEMPTS で gone に落ちる筋も作ってまう)
    _today = datetime.now()
    if date > _today.strftime("%Y%m%d"):
        if verbose:
            print(f"{date}: 未来の日付。飛ばす")
        return {"ok": 0, "empty": 0, "error": 0, "skip": 0, "blocked": 0}
    if date == _today.strftime("%Y%m%d") and _today.hour < FIRST_DATA_HOUR:
        if verbose:
            print(f"{date}: まだ {FIRST_DATA_HOUR}時前。今日の分は在り得んので飛ばす"
                  f"(実測の同日初取得は08:58〜10:32、翌朝に約130本取れる)")
        return {"ok": 0, "empty": 0, "error": 0, "skip": 0, "blocked": 0}
    have = conn.execute("SELECT COUNT(*) FROM oriten_done WHERE date=?", (date,)).fetchone()[0]
    todo = conn.execute(
        "SELECT COUNT(*) FROM oriten_done WHERE date=? "
        "AND status NOT IN ('ok','empty','gone','cancelled')", (date,)).fetchone()[0]
    if have and not todo:
        return {"ok": 0, "empty": 0, "error": 0, "skip": have, "blocked": 0}
    if have:
        # その日の (場, 最大レース番号) は台帳に残っとる。**組み直せるもんを
        # 毎回ネットで列挙し直すな**(列挙の1発が403に当たると150秒待つ)。
        plan = {j: n for j, n in conn.execute(
            "SELECT jcd, MAX(rno) FROM oriten_done WHERE date=? GROUP BY jcd", (date,))}
    else:
        plan = enumerate_day(date)
        time.sleep(SLEEP_SEC)
    if jcds:
        plan = {k: v for k, v in plan.items() if k in jcds}
    counts = {"ok": 0, "empty": 0, "error": 0, "skip": 0, "blocked": 0}
    consec = 0
    if not plan:
        if verbose:
            print(f"{date}: 開催なし (or 列挙失敗)")
        return counts
    probed: set[str] = set()
    for jcd in sorted(plan):
        nmax = plan[jcd]
        for rno in range(1, nmax + 1):
            if _already_done(conn, date, jcd, rno):
                counts["skip"] += 1
                continue
            # 出しとらん場は飛ばす。**ただし1日1本だけ様子見**(目を塞がんため)
            if jcd in SKIP_JCD:
                if jcd in probed:
                    counts["skip"] += 1
                    continue
                probed.add(jcd)
            st = fetch_race(conn, date, jcd, rno)
            counts[st] += 1
            if st == "blocked":
                consec += 1
                if consec >= MAX_CONSEC_BLOCKED:
                    # 🚨 403 が続いた。**弾かれとるんか、ほんまに不在なんか**を
                    #    その場で対照実験して決める:
                    #    台帳で ok と分かっとるレースを1本叩く。
                    #      対照が 200 → 弾かれとらん。403 は**不在**やから、その日を
                    #                   最後まで回して attempts を積む(いずれ gone)
                    #      対照も 403 → **ほんまに弾かれとる**。畳んで出直す
                    conn.commit()
                    if _control_ok(conn):
                        if verbose:
                            print(f"  {date}: {consec}連続403やが対照は200"
                                  f"=不在とみなして続行", flush=True)
                        consec = 0
                    else:
                        if verbose:
                            print(f"  {date}: {consec}連続403で対照も403"
                                  f"=弾かれとる。畳む(次回やり直す)", flush=True)
                        return counts
            else:
                consec = 0
            time.sleep(SLEEP_SEC)
        conn.commit()
        if verbose:
            print(f"  {date} {jcd}{venue_name(jcd)}: "
                  f"ok={counts['ok']} empty={counts['empty']} "
                  f"error={counts['error']} skip={counts['skip']}")
    return counts


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="オリジナル展示データ収集 (BOATCAST)")
    ap.add_argument("--start", required=True, help="開始日 YYYYMMDD")
    ap.add_argument("--end", help="終了日 YYYYMMDD (省略時は --start と同じ)")
    ap.add_argument("--jcd", nargs="*", help="場コード指定 (省略時は全場)")
    ap.add_argument("--quiet", action="store_true")
    # 🚨 毎晩ちょっとずつ取り戻すための上限。
    #    一気に流すと弾かれるし、弾かれた分を追いかけると何時間も張り付く。
    #    **急がん**。blocked/requeue は済み扱いにせんので、毎晩 N 本ずつで必ず収束する。
    ap.add_argument("--max-races", type=int, default=0, help="この本数だけ取ったら止める(0=無制限)")
    args = ap.parse_args()
    end = args.end or args.start
    jcds = [j.zfill(2) for j in args.jcd] if args.jcd else None

    conn = storage.connect()
    _init(conn)
    mark_cancelled_gone(conn, verbose=not args.quiet)
    global SKIP_JCD
    sil = silent_venues(conn)
    if sil and not args.jcd:
        SKIP_JCD = set(sil)
        for j, (n, _ok) in sorted(sil.items()):
            print(f"  出しとらん場として飛ばす: 場{j}({n:,}本試して ok ゼロ)"
                  f" ※1日1本だけ様子見はする", flush=True)
    total = {"ok": 0, "empty": 0, "error": 0, "skip": 0, "blocked": 0}
    for date in daterange(args.start, end):
        c = collect_day(conn, date, jcds, verbose=not args.quiet)
        for k in total:
            total[k] += c.get(k, 0)
        if args.max_races and (total["ok"] + total["blocked"]) >= args.max_races:
            print(f"  上限 {args.max_races} 本に達したんで止める"
                  f"(残りは次回。blocked/requeue は済み扱いにせん)", flush=True)
            break
    left = conn.execute(
        "SELECT COUNT(*) FROM oriten_done "
        "WHERE status NOT IN ('ok','empty','gone','cancelled')").fetchone()[0]
    # 🚨 出しとらん場の分は**もう追わん**ので、素の「未回収」に混ぜたら
    #    **永久に減らん数字を毎晩刷ることになる**。分けて出す。
    #    追う気の無いもんを残件に数えたら、門が鳴っても誰も動かんくなる
    #    → [[insight_a_gate_that_cries_wolf]] / [[insight_zero_is_a_measurement]]
    dead = 0
    if SKIP_JCD:
        ph = ",".join("?" * len(SKIP_JCD))
        dead = conn.execute(
            f"SELECT COUNT(*) FROM oriten_done WHERE status NOT IN ('ok','empty','gone','cancelled')"
            f" AND jcd IN ({ph})", tuple(sorted(SKIP_JCD))).fetchone()[0]
    conn.close()
    tail = (f" / 未回収 {left - dead:,}"
            f"(+ 出しとらん場 {dead:,} は追わん: 場{'・'.join(sorted(SKIP_JCD))})"
            if dead else f" / 未回収 {left:,}")
    print(f"\n完了: ok={total['ok']} empty={total['empty']} blocked={total['blocked']} "
          f"error={total['error']} skip={total['skip']}{tail}")


if __name__ == "__main__":
    main()
