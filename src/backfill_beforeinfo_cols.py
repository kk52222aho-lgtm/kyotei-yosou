"""直前情報ページの生HTMLから、**残されとらんかった列**を通信ゼロで復元する。

## 何が死んどったか(2026-09-26 に数えた)

直前情報が在る期間(20260717〜)の `entries` で、列の埋まりを並べたら
**B/K ファイル由来は全部埋まっとって、直前情報ページ由来だけが全部ゼロ**:

    weight(B由来)      100.0%      tenji_time(K由来)  96.7%
    wind_speed(K由来)   97.9%      wave_height(K由来) 97.9%
    ────────────────────────────────────────────────
    weight_today          0.0%  🚨      tilt            0.0%  🚨
    tenji_st              0.0%  🚨      temperature     0.0%  🚨
    water_temp            0.0%  🚨

🚨 **「配管が切れとる」は誤診やった。**`scraper.fetch_beforeinfo` を実物で叩いたら
6艇とも4列そろって返ってくる。予測(`predict.predict_race`)も毎回それを食っとる。
死んどったんは **取れた値を `entries` に残す書き手**だけで、取得やない。
→ 直し方は「スクレイパを直す」やのうて「派生を日次で作る」(refresh_derived に載せた)

## 生HTMLの形と、3回踏んだ穴

    艇ごと    : 「1 川野 芽唯 47.0kg 6.76 0.0 R 進入」
                = 艇番 / 名前 / **当日体重** / 展示タイム / **チルト**
    レースごと: 「水面気象情報 **7R時点** 気温 20.2℃ 雨 風速 0m 水温 25.4℃ 波高 1cm」

## 🚨 検算の仕込み(2本)

展示タイムは**既に K ファイル由来で96.7%埋まっとる**。同じ行から解いた展示タイムを
それと突き合わせれば、**隣の列(体重・チルト)の解き方が正しいか**が分かる。
気象も同じで、気温と水温には真値が無いが**同じ1文の風速と波高には在る**。
独立な出どころで測る検算やから、grep で自分を確かめるんとはちがう。
→ [[insight_verifier_shares_the_blind_spot]]

門が実際に3回止めた: 81.38%(艇番を前の行から拾っとった)/ 42.12%(下記)/ 91.89%。

## 🚨 気象は entries に書かん。**別の量やから**

ページに刻まれとるんは「**7R時点**」——R8のページに載っとるんは R7 時点の観測値で、
当該レースの気象やない(R8はまだ走っとらん。当たり前や)。解けた6,411レースの
**100.0%**に刻印が在って、番号は全部当該レースとちがう。遅れは **1レース前 2,727 /
2レース前 2,857**(=20〜40分前)。刻印レースのK気象と突き合わせたら **5,456/5,456
= 100.00%**、当該レースと突き合わせたら **41.01%**。

せやから `temperature`/`water_temp` は**死んだまま残す**。列名が時点を表現でけへんから、
書いたら次に読む人が当該レースの気温と読む。住処は `beforeinfo_weather` で、
**at_rno(何レース時点か)を必ず持って回る**。

## 🚨 ついでに出た: `wind_speed`/`wave_height` は先読みやが決定を動かさん

`features.py` の特徴量に入っとるこの2列はK由来=**レース中の観測**で、締切時点では
知れん。締切前に知れる値との差は 風速|差|中央 **1.00m**・56.1%のレースで1m以上・
16.0%で2m以上。**せやが平均差は +0.030 で系統的な偏りが無い**。凍結モデルに両方
食わせて突き合わせたら(5,119レース):

    本命が入れ替わったレース   36/5,119 = 0.70%
    本命的中  K気象(いま)      57.687%
    本命的中  締切前の気象       57.570%   差 +0.117pt(二項の目安 sd も 0.117pt)

入れ替わりが 0.70% しか無いんやから**検出力不足やのうて、ほんまに動かん**。
看板(AUC 0.83 / 本命57.7%)は無傷。学習と本番で別の値を食っとる(train-serving
skew)のは事実やが、大きさはここに書いた通り。
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata

# 「1 川野 芽唯 47.0kg 6.76 0.0」= 艇番 / 名前 / 当日体重 / 展示タイム / チルト
#
# 🚨 最初の版は `([1-6])\s+\S[^\d]*?(\d{2}\.\d)kg` で名前を最短一致で飛ばしとった。
#    それやと **前の艇の末尾から艇番を拾う**:
#      「…着順 3 5 松瀬 弘美 46.5kg」→ 3(前の艇の着順)を艇番と読む
#    検算(K由来の展示タイムとの突合)が **81.38%** で門に止められた。
#    直し: **艇番と kg の間に数字が1つも無い**ことを要求する。
#    そうすると 3 は「5 松瀬」を跨がなあかんくなって落ちる。
#    名前は1〜4語に限る(暴走止め)。
#
# 🚨 2度目の直し: 数字を**3つ**要求しとったんで、展示タイムがまだ出てへん行が
#    丸ごと落ちて解け率が70%で止まっとった。
#      解ける : 1 川野 芽唯 47.0kg 6.76 0.0 R   ← 体重・展示タイム・チルト
#      解けん : 1 永田 秀二 55.5kg      0.5 R   ← 展示タイムが**まだ無い**
#    撮った時点は全部「締切15分前」で揃っとるが、**展示走行の時刻は場ごとにちがう**
#    (場03 は 23.2% / 場02 は 94.5% で割れた)。体重とチルトは在る。
#    → 後ろを `R`(プロペラ欄)で止めて、間の数字の**個数**で読み分ける。
#
# 🚨 3度目の直し: 解け率は 70%→98.9% に上がったが **6艇そろわんレースが 43→1,768**。
#    「解けた数が増えた」だけ見とったら見落とす([[insight_zero_is_the_best_looking_number]])。
#    原因は**部品交換の語がチルトと R の間に挟まる**ことで、単一やった:
#      2 柳瀬 興志 58.9kg 6.71 -0.5 キャブ R
#      5 中北 将史 52.8kg 0.0 リング×2 R      ← NFKC後の ×2 の 2 も邪魔しとった
#    → 数字列と R の間に「数字で始まらんトークン」を4個まで許す。
#    ついでに数字が**0個**(体重だけで展示もチルトも未発表)も通す。
#    数字が1個の時は**値で読み分ける**: チルトは -1.0〜3.0 やから 5.0 以上は展示タイム。
#    推測せんでも桁で判る量を推測で埋めん。
LANE_ROW = re.compile(
    r"([1-6])((?:\s+[^\d\s]+){1,4})\s+(\d{2}\.\d)kg((?:\s+-?[0-9.]+){0,2})"
    # 部品交換は最大で5語見た(ピストン×2 リング×4 電気 シリンダ シャフト)。
    # 9語まで許す。数字で始まるトークンは許さんので次の艇番に届く前に必ず止まる
    r"((?:\s+[^\d\s]\S*){0,9})\s+(?:R|新)(?:\s|$)")
# 「気温 20.2℃ 雨 風速 0m 水温 25.4℃ 波高 1cm」
#
# 🚨 `℃` を要求しとったら **1件も当たらんかった**。NFKC が ℃ を `°C` の2文字に
#    分解するからや。しかも COALESCE で上書きを避ける作りにしとったんで
#    **書き込みが「成功」して気象2列だけ 0.0% のまま通った**。
#    門は展示タイムしか見とらんかったから気象側は無検算やった
#    → [[insight_missing_reads_as_zero]]。風速・波高でも検算する(下)
# 🚨 「水面気象情報 **7R時点**」。R8のページに載っとるんは R7 時点の観測値で、
#    **当該レースの気象やない**(R8はまだ走っとらん。当たり前や)。
#    刻印は解けた6,411レースの**100.0%**に在って、番号は全部当該レースとちがう。
AT_RNO = re.compile(r"水面気象情報\s*(\d{1,2})R時点")
COND = re.compile(
    r"気温\s*([\d.]+)\s*(?:℃|°C)\s*(\S*?)\s*風速\s*([\d.]+)m"
    r"\s*水温\s*([\d.]+)\s*(?:℃|°C)\s*波高\s*([\d.]+)cm")


WX_DDL = """
CREATE TABLE IF NOT EXISTS beforeinfo_weather (
  date TEXT, jcd TEXT, rno INTEGER,   -- どのレースのページに載っとったか
  at_rno INTEGER,                     -- 🚨 **何レース時点の観測か**(ほぼ必ず rno-1)
  temperature REAL, water_temp REAL, wind_speed REAL, wave_height REAL,
  weather TEXT,
  PRIMARY KEY (date, jcd, rno)
);
"""


def parse_raw(raw: str) -> tuple[dict[int, dict], dict]:
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", raw or ""))
    per: dict[int, dict] = {}
    for lane, _name, w, nums, _parts in LANE_ROW.findall(t):
        ln = int(lane)
        if ln in per:            # 同じ艇番が2回出たら後を信じん(表が崩れとる合図)
            continue
        vs = [float(x) for x in nums.split()]
        if len(vs) == 2:
            tenji, tilt = vs[0], vs[1]
        elif len(vs) == 1:
            # 展示タイムは 6.xx 台、チルトは -1.0〜3.0。桁で分かれるんで桁で読む
            tenji, tilt = (vs[0], None) if vs[0] >= 5.0 else (None, vs[0])
        else:
            tenji = tilt = None  # 体重だけ出とる(展示前)
        per[ln] = {"weight_today": float(w), "tenji_time": tenji, "tilt": tilt}
    cond: dict = {}
    m = COND.search(t)
    if m:
        ar = AT_RNO.search(t)
        cond = {"temperature": float(m.group(1)), "water_temp": float(m.group(4)),
                # 🚨 使わん2つも返す。**K由来の真値と突き合わせる検算用**や
                "wind_speed": float(m.group(3)), "wave_height": float(m.group(5)),
                "weather": m.group(2) or "",
                # 🚨 **何レース時点の観測か**を必ず持って回る。これを落としたら
                #    「当該レースの気象」に化けて先読みと区別でけへんくなる
                "at_rno": int(ar.group(1)) if ar else None}
    return per, cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/kyotei.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")

    rows = con.execute(
        "SELECT date, jcd, rno, raw_text FROM beforeinfo_raw "
        "WHERE raw_text IS NOT NULL").fetchall()
    n_race = n_lane = 0
    agree = miss = 0
    bad_lane = empty_page = unparsed = 0
    n_tilt = n_tenji = 0
    n_cond = c_ok = c_tot = c_nostamp = 0
    upd: list[tuple] = []
    wx: list[tuple] = []
    for date, jcd, rno, raw in rows:
        per, cond = parse_raw(raw)
        if not per:
            # 🚨 「解けんかった」と「向こうが出しとらん」を分ける。
            #    体重(kg)が1つも無いページは直前情報が**まだ何も出てへん**ので
            #    これは解き方の問題やない。混ぜたら道具の欠陥が隠れる。
            if "kg" in unicodedata.normalize("NFKC", raw or ""):
                unparsed += 1
            else:
                empty_page += 1
            continue
        if len(per) != 6:
            bad_lane += 1
        n_race += 1
        # 🚨 検算その2: 気象。解いた**風速と波高**を K 由来の値と突き合わせる。
        #    気温と水温には真値が無いが、同じ1文から同じ解き方で取った隣の2つに
        #    真値が在る。ここが合えば気温・水温の解き方も信じてええ
        if cond:
            n_cond += 1
            kw = con.execute(
                "SELECT wind_speed, wave_height FROM entries WHERE date=? AND jcd=?"
                " AND rno=? AND wind_speed IS NOT NULL LIMIT 1",
                # 🚨 突合は**刻印されたレース番号**で取る。当該レース番号で当てたら
                #    41.01% しか合わん——それは解き方が悪いんやのうて**別の時点**やから
                (date, str(jcd).zfill(2), cond["at_rno"] or -1)).fetchone()
            if cond["at_rno"] is None:
                # R1・R2 は**その日まだ観測が無い**(前レースが走っとらん)。
                # 検算でけへんもんを分母に入れたら門が狼少年になる
                c_nostamp += 1
            elif kw:
                c_tot += 1
                c_ok += (abs(kw[0] - cond["wind_speed"]) < 0.51
                         and abs(kw[1] - cond["wave_height"]) < 0.51)
            wx.append((date, str(jcd).zfill(2), int(rno), cond["at_rno"],
                       cond["temperature"], cond["water_temp"],
                       cond["wind_speed"], cond["wave_height"], cond["weather"]))
        # 🚨 検算: 解いた展示タイムを K 由来の値と突き合わせる
        known = {int(l): t for l, t in con.execute(
            "SELECT lane, tenji_time FROM entries WHERE date=? AND jcd=? AND rno=?"
            " AND tenji_time IS NOT NULL", (date, str(jcd).zfill(2), int(rno)))}
        for ln, v in per.items():
            n_lane += 1
            n_tilt += v["tilt"] is not None
            n_tenji += v["tenji_time"] is not None
            if ln in known and v["tenji_time"] is not None:
                if abs(known[ln] - v["tenji_time"]) < 0.005:
                    agree += 1
                else:
                    miss += 1
            upd.append((v["weight_today"], v["tilt"],
                        date, str(jcd).zfill(2), int(rno), ln))

    # 🚨 気象の門。ここを付けるまで気象2列は**無検算で 0.0% のまま「成功」**しとった
    if n_cond:
        r = c_ok / c_tot if c_tot else 0.0
        print(f"  気象が解けたレース: {n_cond:,}/{n_race:,} = {n_cond/n_race:.1%}")
        print(f"  刻印が無い(=まだ観測が無い R1・R2): {c_nostamp:,}  ← 検算不能。分母に入れん")
        print(f"  🚨 検算(解いた風速・波高 vs 刻印レースのK由来): 一致 {c_ok:,}/{c_tot:,}"
              f" = {r:.2%}" if c_tot else "  気象の検算でける行が無い")
        if c_tot and r < 0.95:
            print("  🚨 気象の一致率が95%未満。**解き方が怪しいんで書かん。**")
            return
        if n_cond / n_race < 0.90:
            print("  🚨 気象が9割のレースで解けとらん。**書かん。**")
            return
    else:
        print("  🚨 気象が1レースも解けとらん。**書かん。**")
        return
    tot = agree + miss
    print(f"raw_text {len(rows):,}レース → 解けた {n_race:,}レース / {n_lane:,}艇")
    print(f"  ページに直前情報が1つも出てへん(kg無し): {empty_page}  ← 欠損。解き方の話やない")
    mk = "  🚨" if unparsed else ""
    print(f"  kgは在るのに解けんかった: {unparsed}{mk}")
    print(f"  6艇そろわんレース: {bad_lane}")
    print(f"  艇ごとの埋まり: 体重 {n_lane:,} / チルト {n_tilt:,} ({n_tilt/n_lane:.1%})"
          f" / 展示タイム {n_tenji:,} ({n_tenji/n_lane:.1%})")
    print(f"  🚨 検算(解いた展示タイム vs K由来): 一致 {agree:,} / 不一致 {miss:,}"
          f" = {agree/tot:.2%}" if tot else "  検算でける行が無い")
    if tot and agree / tot < 0.99:
        print("  🚨 一致率が99%未満。**解き方が怪しいんで書かん。**")
        return
    if a.dry_run:
        print("  (dry-run。書かん)")
        return
    # 🚨 COALESCE。解けんかった列に NULL を上書きして**既に在る値を消さん**。
    #    いまは4列とも0%やから無害やが、前向きの配管が直った後に再実行しても
    #    壊れん形にしとく([[insight_missing_reads_as_zero]]の裏)。
    con.executescript(WX_DDL)
    con.executemany("INSERT OR REPLACE INTO beforeinfo_weather VALUES (?,?,?,?,?,?,?,?,?)", wx)
    con.executemany(
        "UPDATE entries SET weight_today=COALESCE(?,weight_today),"
        " tilt=COALESCE(?,tilt)"
        " WHERE date=? AND jcd=? AND rno=? AND lane=?", upd)
    con.commit()
    d0, d1 = con.execute("SELECT MIN(date), MAX(date) FROM beforeinfo_raw").fetchone()
    n = con.execute("SELECT COUNT(*) FROM entries WHERE date BETWEEN ? AND ?",
                    (d0, d1)).fetchone()[0]
    print("  書いた後の埋まり(直前情報が在る期間):")
    for col in ("weight_today", "tilt"):
        f = con.execute(f"SELECT SUM({col} IS NOT NULL) FROM entries"
                        " WHERE date BETWEEN ? AND ?", (d0, d1)).fetchone()[0]
        print(f"    {col:14s} {f:7,}/{n:,} = {f/n:6.1%}")


if __name__ == "__main__":
    main()
