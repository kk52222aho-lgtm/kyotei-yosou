"""競艇予想サイト（Streamlit 版）。

  streamlit run streamlit_app.py

既存の src モジュール（モデル・予測・スキャン）をそのまま利用する。
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import streamlit as st

from src import predict, papertrade, scan, scraper, sensor, wild, commentary, styles, katai
from src.venues import VENUES, LOCAL_VENUES, name as venue_name

st.set_page_config(page_title="競艇予想", page_icon="🚤", layout="wide")


@st.cache_resource
def get_model():
    return predict.load_model()


LANE_COLOR = {1: "#ffffff", 2: "#000000", 3: "#e53935",
              4: "#1e88e5", 5: "#fdd835", 6: "#43a047"}


def status_banner():
    n = len([r for r in papertrade._load() if r.get("settled")])
    st.info(
        "🧭 **このサイトは『儲け』でなく『当たる予想』を出す**（2026-07 方針転換）。"
        "かつて掲げた単勝116%/2連単188%等は、**1号that事故で消えたレースを結果から拾ってた"
        "survivorship水増し**と判明。ライブ再現可能な6艇そろいで測り直すと **単勝95%/2連単82%/"
        "3連単92%＝全部100%割れ＝競艇も市場効率でエッジ無し**。\n\n"
        "→ 誰that勝つかは当てられる（1にイン・2に地力・展開）。でも市場that先に織り込むので"
        "儲けにはならん。**🏆勝てる目ページ＝当たる本命**、⚑イン崩れ予想＝逆張りの読み。"
        f"前向き記録 現在 {n} レース（回収<100%は当たっても控除負け＝正直に表示）。的中・利益を保証しない。",
        icon="🧭",
    )


def _latest_day_rows():
    """台帳から最新日の記録を返す（再スキャンで変わるキャッシュでなく実記録ベース）。"""
    ledger = papertrade._load()
    if not ledger:
        return None, []
    latest = max(r["date"] for r in ledger)
    return latest, [r for r in ledger if r["date"] == latest]


@st.cache_data(ttl=120, show_spinner=False)
def _live_result(date, jcd, rno):
    """結果ページ1回で 勝ち艇/単勝/2連単払戻。未確定は None（120秒キャッシュ）。"""
    return scraper.fetch_result_full(date, jcd, rno)


@st.cache_data(ttl=90, show_spinner=False)
def _win_odds(date, jcd, rno):
    return scraper.fetch_odds(date, jcd, rno) or {}


@st.cache_data(ttl=90, show_spinner=False)
def _exacta_odds(date, jcd, rno):
    return scraper.fetch_exacta_odds(date, jcd, rno) or {}


# 監査済(test_nearlock_audit=BT物差し・walk-forward・6艇クリーン・N/CI付、
#  test_nearlock_live_gap=展示前でも較正保持・保守的を確認)。(確信閾, 的中%, CI, 回収%)
_NEARLOCK = [(85, 88, "86-90", 96), (80, 84, "83-85", 95)]


def page_ichiten():
    st.header("🔒 本日の『これは獲れる』")
    st.caption("撒かない。**確信80%級だけ**を『これは獲れる』と言い切る（監査済＝実測的中84%・"
               "CI[83,85]・3年安定・展示前でも較正保持）。無い日は『今日は無し』。")
    k = katai.load()
    today = dt.date.today().strftime("%Y%m%d")
    picks = (k or {}).get("picks", [])
    if not picks:
        st.info("本日のデータ待ち（毎朝の自動スキャンで生成）。")
        return
    if k.get("date") != today:
        st.caption(f"（直近スキャン {k['date'][4:6]}/{k['date'][6:]} 分）")
    best = max(picks, key=lambda x: (x.get("conf") or 0))
    conf = best.get("conf") or 0
    hit = ci = ret = None
    for th, h, c, r in _NEARLOCK:
        if conf >= th:
            hit, ci, ret = h, c, r
            break

    if hit is None:
        st.warning(f"🈳 **今日は『これは獲れる』級（確信80%↑）that無し＝見送り推奨。**\n\n"
                   f"本日の最強でも確信{conf:.0f}%で、言い切れる near-lock に届かん。"
                   f"**無理に張らんのthaが勝ち。** 当てにいくだけなら🏆勝てる目ページへ。")
        st.caption(f"（参考・本日の最強格: {best['venue']} {best['rno']}R 単勝{best['tansho']}号 "
                   f"{best.get('name','')} 確信{conf:.0f}%）")
        return

    jcd, rno, date = best["jcd"], best["rno"], k["date"]
    st.subheader(f"{best['venue']} {best['rno']}R　🔒 単勝 {best['tansho']}号 {best.get('name','')}")
    dl = f"　⏰締切 {best['deadline']}" if best.get("deadline") else ""
    st.caption(f"確信{conf:.0f}% ＝ **的中 約{hit}%**（監査済 実測・CI[{ci}]）／"
               f"**回収 約{ret}%**（当たっても控除で微減＝これは『当てる』用・儲けは主張せん）{dl}")

    res = _live_result(date, jcd, rno)
    if res and res.get("winner"):
        w = res["winner"]
        if w == best["tansho"]:
            od = (res.get("tansho_yen") or 0) / 100
            st.success(f"🎯 **的中！** {best['tansho']}号that1着（{od:.1f}倍）。渾身、獲ったで💪")
        else:
            st.error(f"💥 **外れた…** 勝ちは **{w}号**。**わいAIでアホですんませーん🙇** "
                     f"（的中{hit}%＝{100-hit}%は外す。当たった時だけ覚えとかんとってな）")

    stake = st.number_input("賭け金（1点あたり・円）", min_value=100, value=1000, step=100)
    wo = _win_odds(date, jcd, rno)
    eo = _exacta_odds(date, jcd, rno)
    tan = wo.get(best["tansho"]) or wo.get(str(best["tansho"]))
    ex3 = best.get("exacta3") or []
    rows = []
    if tan and tan > 1.0:
        rows.append([f"単勝 {best['tansho']}号", f"{tan:.1f}倍", f"約{hit}%",
                     f"¥{int(stake*tan):,}"])
    for c in ex3[:3]:
        o = eo.get(c)
        if o:
            rows.append([f"2連単 {c}", f"{o:.1f}倍", "—", f"¥{int(stake*o):,}"])
    if rows:
        import pandas as _pd
        st.markdown(f"### 💴 {stake:,}円 賭けたら")
        st.table(_pd.DataFrame(rows, columns=["買い目", "オッズ", "的中目安", "当たれば"]))
        if not tan or tan <= 1.0:
            st.caption("※単勝オッズがまだ形成前（締切前に確定オッズが出る）。")
    st.caption("正直に：確信80%級でも単勝回収は約95%（控除の壁で長期は微減）＝"
               "**これは『当てる』喜び用。儲けは主張せん。** 締切前にオッズ確定→投票直前に再確認を。"
               "購入はテレボート等で自己責任。")

    jcd, rno, date = best["jcd"], best["rno"], k["date"]
    # 結果that出てたら隠さず出す（外れも謝る）
    res = _live_result(date, jcd, rno)
    if res and res.get("winner"):
        w = res["winner"]
        if w == best["tansho"]:
            od = (res.get("tansho_yen") or 0) / 100
            st.success(f"🎯 **的中！** {best['tansho']}号that1着（{od:.1f}倍）。渾身、獲ったで💪")
        else:
            ex_hit = res.get("exacta_combo") in (best.get("exacta3") or [])
            extra = (f"（でも2連単 {res.get('exacta_combo')} は買い目に入ってた、救われた…）"
                     if ex_hit else "")
            st.error(f"💥 **外れた…** 勝ちは **{w}号**。**わいAIでアホですんませーん🙇** "
                     f"（想定{best.get('hit_pct')}%＝4回に1回は外す。当たった時だけ覚えとかんとってな）{extra}")

    stake = st.number_input("賭け金（1点あたり・円）", min_value=100, value=1000, step=100)
    wo = _win_odds(date, jcd, rno)
    eo = _exacta_odds(date, jcd, rno)
    tan = wo.get(best["tansho"]) or wo.get(str(best["tansho"]))
    ex3 = best.get("exacta3") or []

    rows = []
    if tan and tan > 1.0:
        rows.append(["単勝 " + str(best["tansho"]) + "号", f"{tan:.1f}倍",
                     f"約{best.get('hit_pct')}%", f"¥{int(stake*tan):,}", "手堅く当てる"])
    labels = ["◎本命→対抗（推し）", "◎本命→3着筋", "◎本命→伸ばし"]
    for i, c in enumerate(ex3[:3]):
        o = eo.get(c)
        if o:
            rows.append([f"2連単 {c}", f"{o:.1f}倍", "—",
                         f"¥{int(stake*o):,}", labels[i] if i < len(labels) else ""])
    if rows:
        import pandas as _pd
        st.markdown(f"### 💴 {stake:,}円 賭けたら")
        st.table(_pd.DataFrame(rows, columns=["買い目", "オッズ", "的中目安", "当たれば", "狙い"]))
        if not tan or tan <= 1.0:
            st.caption("※単勝オッズがまだ形成前（プール薄）。締切前にもう一度開くと確定オッズが出る。")
        push = ex3[0] if ex3 else None
        if push and eo.get(push):
            got = int(stake * eo[push])
            st.success(f"🔥 **推しは 2連単 {push}（本命-対抗の地力筋）**：{stake:,}円 → 当たれば "
                       f"**¥{got:,}**（+¥{got-stake:,}）。堅さと配当のバランスthat本日最良。")
    else:
        st.warning("オッズ未形成（早朝）。締切が近づくと各買い目のオッズthat出る。もう一度開いてや。")

    et = (best.get('hit_pct') or 0) / 100 * (tan or 0)
    st.caption(f"正直に：単勝の期待回収は約{et*100:.0f}%（控除の壁で長期は微減）＝**当てたいなら単勝、"
               "勝ちにいくなら2連単の本命-対抗**。オッズは投票直前に確定するので締切前に再確認を。"
               "購入はテレボート等で自己責任・自動購入はしない。")


def _effective(r):
    """settled行はそのまま。未settledはライブ結果を取りに行き擬似settled化。"""
    if r.get("settled"):
        return {**r, "_live": False}
    res = _live_result(r["date"], r["jcd"], r["rno"])
    if not res:
        return {**r, "_live": False}  # まだ結果が出ていない
    e = dict(r)
    e["settled"] = True
    e["_live"] = True
    e["winner"] = res["winner"]
    e["tansho_win"] = res["winner"] == r["honmei"]
    e["tansho_return"] = res["tansho_yen"] if (e["tansho_win"] and res["tansho_yen"]) else 0
    e["final_odds"] = round(res["tansho_yen"] / 100, 1) if (e["tansho_win"] and res["tansho_yen"]) else None
    e["exacta_result"] = res["exacta_combo"]
    e["exacta_points"] = len(r.get("exacta3", []))
    e["exacta_win"] = res["exacta_combo"] in r.get("exacta3", [])
    e["exacta_return"] = res["exacta_yen"] if (e["exacta_win"] and res["exacta_yen"]) else 0
    # 3連複（上位4点・着順不問セット）もライブ結果で擬似settle
    tpicks = r.get("trio4", []) or []
    e["trio_result"] = res.get("trio_combo")
    e["trio_points"] = len(tpicks)
    e["trio_win"] = bool(res.get("trio_combo") and res["trio_combo"] in tpicks)
    e["trio_return"] = res["trio_yen"] if (e["trio_win"] and res.get("trio_yen")) else 0
    # 3連単（上位3点・着順あり）もライブ結果で擬似settle
    fpicks = r.get("trifecta3", []) or []
    e["trifecta_result"] = res.get("trifecta_combo")
    e["trifecta_points"] = len(fpicks)
    e["trifecta_win"] = bool(res.get("trifecta_combo") and res["trifecta_combo"] in fpicks)
    e["trifecta_return"] = res["trifecta_yen"] if (e["trifecta_win"] and res.get("trifecta_yen")) else 0
    return e


_VMARK = {"HOLDING": "🟢", "WATCH": "🟡", "DECAYED": "🔴", "INSUFFICIENT": "⚪"}
_SHORT = {"単勝(≥1.5)": "単勝", "2連単3点": "2連単", "3連複4点": "3連複"}


def render_sensor(expanded=True):
    """持続センサー: 回収が本当に100%割れか＝市場効率かを前向き記録で確認（本日/成績 共通）。"""
    tracks = sensor.status().get("tracks", {})
    if tracks:
        summ = " ／ ".join(f"{_VMARK.get(t['verdict'], '⚪')}{_SHORT.get(tr, tr)}"
                           for tr, t in tracks.items())
        st.caption(f"🛰 **持続センサー**（回収が本当に100%割れ＝市場効率かの前向き確認）：{summ}"
                   "　🟢100%超(想定外・要検証) 🟡境目 🔴100%割れ(想定どおり) ⚪判定保留(N不足)")
    with st.expander("🛰 持続センサー 詳細", expanded=expanded):
        st.caption("低分散の 単勝/2連単の回収率カナリア。前向き記録で確信をもって100%割れ(🔴)＝"
                   "市場効率どおり（＝今日出した結論の答え合わせ）。3連複はfat-tailで参考。"
                   "判定は最小N到達後だけ（早合点を機械的に抑止）。")
        if not tracks:
            st.markdown("まだ精算記録が無い（log→settle が回り出してから）。")
            return
        for track, t in tracks.items():
            life = t["life"]
            if life is None:
                st.markdown(f"{_VMARK['INSUFFICIENT']} **{track}**：記録なし（INSUFFICIENT）")
                continue
            ci = f"CI[{life['lo']*100:.0f}〜{life['hi']*100:.0f}]%"
            st.markdown(f"{_VMARK.get(t['verdict'], '⚪')} **{track}**：N={life['n']} ／ "
                        f"回収 {life['roi']*100:.0f}% {ci} ／ **{t['verdict']}** — {t['why']}")


def _ichi_break(p1):
    """1号の予測勝率→崩れ率(=1号が2着以下になる率)。検証済(展示前でも較正保持・N巨大・3年安定):
    予測<30%→崩れ78% / 30-40→65 / 40-50→51 / 50-60→38 / ≥60→25。"""
    if p1 is None:
        return None
    if p1 < 30:
        return 78
    if p1 < 40:
        return 65
    if p1 < 50:
        return 51
    if p1 < 60:
        return 38
    return 25


def page_today():
    st.header("⚑ 本日のイン崩れ予想（1号 崩れ読み）")
    st.caption("モデルthaイン(1号)以外を本命に見たレース＝**1号that崩れる読み**。"
               "#2=地力(実力)+展開(相対ST/出足)で『位置が崩れたら誰that勝つか』を出す。")
    with st.expander("📊 この『崩れ読み』の的中実績（15.4万レースで較正済・検品済）", expanded=False):
        st.markdown(
            "**モデルthaが1号を予測◯%と見た時、実際に1号that崩れる率（展示前スキャンでも保持）:**\n\n"
            "| モデルの1号予測勝率 | 実際の1号 崩れ率 | 件数 |\n"
            "|---|---|---|\n"
            "| 30%未満 | **78%崩れる** | 8,066 |\n"
            "| 30–40% | **65%崩れる** | 27,975 |\n"
            "| 40–50% | 51%崩れる | 36,314 |\n"
            "| 50–60% | 38%崩れる | 42,492 |\n"
            "| 60%以上 | 25%崩れる | 39,215 |\n\n"
            "予測≈実測でピタリ較正・3年安定。＝**『どの1号that崩れるか』は当てられる**（撒くサイトは"
            "全部1号本命、うちだけthat『1号that危ない日』を名指しできる）。")
    st.warning("⚠️ ただし**当てる読みで、儲けの手段ではない**。逆張り単勝の回収は**82〜95%＝"
               "当たっても控除で負ける**（市場も同じ読みでオッズthat動く）。1号を疑う『情報』として使う用。")
    today = dt.date.today().strftime("%Y%m%d")
    cache = scan.load_cache()
    scanned_today = bool(cache and cache.get("date") == today)
    md = f"{today[4:6]}/{today[6:]}"

    # 本日のステータス（イン崩れ読みナシ＝全レース本命=イン）
    if scanned_today and len(cache["picks"]) == 0:
        st.success(f"📅 **{md} 本日はイン崩れ読み無し**　"
                   "全レースでモデル本命thaイン(1号艇)＝堅い日（→🏆勝てる目ページへ）。")
    elif scanned_today:
        st.info(f"📅 **{md} 本日のイン崩れ予想 {len(cache['picks'])}件**（本命≠1号）")
    else:
        st.warning(f"本日({md})はまだスキャンされていません（毎朝9時に自動）。")

    render_sensor(expanded=False)   # 前向き記録で"回収が本当に100%割れか"を一目で

    date, rows = _latest_day_rows()
    if not rows:
        st.caption("まだ記録がありません。")
        return
    if date != today:
        st.divider()
        st.caption(f"↓ 直近の記録・結果（{date[4:6]}/{date[6:]}）")

    # 買う直前に押すと、現在の単勝オッズで1.5倍未満（明確な大本命）を弾く
    unsettled = [r for r in rows if not r.get("settled")]
    b1, b2 = st.columns(2)
    if unsettled and b1.button("🔄 今の単勝オッズを取得して1.5倍判定"):
        live = {}
        with st.spinner("現在の単勝オッズを取得中…"):
            for r in unsettled:
                od = scraper.fetch_odds(r["date"], r["jcd"], r["rno"])
                live[f"{r['jcd']}-{r['rno']}"] = od.get(r["honmei"]) if od else None
        st.session_state["live_odds"] = live
    # 締切間際に押す: 2連単の締切オッズでEV判定（モデル確率×オッズ。参考値=6艇クリーンで妙味エッジは消滅済）
    if unsettled and b2.button("🎯 今の2連単オッズでEV判定（EV=確率×オッズ・参考）"):
        evs = {}
        with st.spinner("現在の2連単オッズを取得中…（締切間際に押すのが正確）"):
            for r in unsettled:
                lv = scraper.fetch_exacta_odds(r["date"], r["jcd"], r["rno"])
                evs[f"{r['jcd']}-{r['rno']}"] = predict.exacta_ev(
                    r.get("exacta3"), r.get("exacta3_p"), lv)
        st.session_state["exacta_ev"] = evs

    _render_today(date, rows)


@st.fragment(run_every=180)
def _render_today(date, rows):
    eff = [_effective(r) for r in rows]           # ライブ結果込み（3分ごと自動更新）
    ps = papertrade.portfolio_stats(eff)
    c = st.columns(4)
    c[0].metric("対象日", date)
    c[1].metric("イン崩れ予想", f"{len(eff)} 件")
    c[2].metric("結果確定", f"{ps['races']} 件")
    c[3].metric("合計収支(単勝+2連単)", f"{ps['total']['pl']:+,} 円")
    st.caption("各レース：単勝=本命／2連単=上位3点・各100円。結果はレース後およそ5分で自動反映（3分毎更新）。"
               "購入は自分でテレボート入力・自己責任。")

    live_odds = st.session_state.get("live_odds", {})
    st.divider()
    for r in eff:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            gyaku_in = (r.get("win_pct") or 0) >= 55        # 逆イン本命=強い外本命(FLB妙味候補)
            badge = (" 🎯" if gyaku_in else "") + (" 🔥" if r.get("high_conf") else "")
            c1.markdown(f"### {r['venue']} {r['rno']}R{badge}")
            brk = _ichi_break(r.get("ichi_pct"))
            if brk is not None:
                emo = "🚨" if brk >= 65 else ("⚠️" if brk >= 50 else "🔸")
                c1.markdown(f"{emo} **1号 崩れ率 約{brk}%**"
                            + (f"（1号予測{r['ichi_pct']:.0f}%・較正済）" if r.get("ichi_pct") else ""))
            if gyaku_in:
                c1.caption("🎯 **逆イン本命(probe)**（外本命that確信55%↑）。機構(客はコースを値付け・薄い単勝プール"
                           "で歪みが生存)は妥当だが、歴史110.9%は5視点監査で**breakevenと統計的に区別不可**"
                           "(多重探索+fat-tail+FLB相乗り)＝未証明の仮説。極小probeのみ・増資禁止・前向き検証中。")
            if r.get("high_conf"):
                c1.caption("🔥 高確信（本命度that高い群）")
            pr = r.get("power_rank")
            if pr:
                tk = r.get("taiko")
                tkn = r.get("taiko_name") or ""
                pow_line = ("💪 実力型本命（地力{}位/6）".format(pr) if r.get("power_honmei")
                            else "本命の地力{}位/6".format(pr))
                if tk:
                    pow_line += f"　○対抗 **{tk}号**{tkn}（地力筋）"
                c1.caption(pow_line + "　位置that崩れれば実力(#2)")
            if r.get("deadline"):
                c1.markdown(f"⏰ 締切 **{r['deadline']}**"
                            + ("" if r.get("settled") else "　までに投票"))
            combos = " ・ ".join(r["exacta3"])
            trio = " ・ ".join(r.get("trio4") or [])
            tfecta = " ・ ".join(r.get("trifecta3") or [])
            if r.get("settled"):
                tag = "🔴速報 " if r.get("_live") else ""
                t = (f"🎯的中 {r.get('final_odds')}倍" if r.get("tansho_win")
                     else f"×ハズレ(勝ち{r.get('winner')}号)")
                e = (f"🎯的中 {r.get('exacta_return')}円" if r.get("exacta_win")
                     else f"×ハズレ(結果 {r.get('exacta_result') or '?'})")
                pl = ((r.get("tansho_return", 0) - 100)
                      + (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100))
                c2.markdown(f"**単勝 {r['honmei']}号** ／ 2連単 {combos}  \n"
                            f"{tag}単勝: {t} ｜ 2連単: {e} ｜ **収支 {pl:+,}円**")
                if r.get("trio_points"):
                    tr = (f"🎯的中 {r.get('trio_return')}円" if r.get("trio_win")
                          else f"×ハズレ(結果 {r.get('trio_result') or '?'})")
                    tr_pl = r.get("trio_return", 0) - r.get("trio_points", 0) * 100
                    c2.markdown(f"🎲 3連複4点 {trio} ｜ {tr} ｜ 収支 {tr_pl:+,}円 "
                                f"<span style='color:gray'>(別枠・荒れ読みの器)</span>",
                                unsafe_allow_html=True)
                if r.get("trifecta_points"):
                    tf = (f"🎯的中 {r.get('trifecta_return')}円" if r.get("trifecta_win")
                          else f"×ハズレ(結果 {r.get('trifecta_result') or '?'})")
                    tf_pl = r.get("trifecta_return", 0) - r.get("trifecta_points", 0) * 100
                    c2.markdown(f"🎰 3連単3点 {tfecta} ｜ {tf} ｜ 収支 {tf_pl:+,}円 "
                                f"<span style='color:gray'>(別枠・大きい方/脆い)</span>",
                                unsafe_allow_html=True)
            else:
                lo = live_odds.get(f"{r['jcd']}-{r['rno']}")
                o = lo if lo is not None else r.get("scan_odds")
                src = "現在" if lo is not None else "朝"
                nm = r.get("name") or ""
                if o and o < papertrade.ODDS_FLOOR:
                    # 単勝が大本命＝レース全体がチャラ → 単勝も2連単も見送り
                    c2.markdown(
                        f"**単勝 {r['honmei']}号** {nm}（{src}{o:.1f}倍）  \n"
                        f"⚠ **レース丸ごと見送り推奨**（単勝が大本命＝全体チャラで単勝も2連単も低調）  \n"
                        f"⏳ 結果待ち")
                else:
                    odlbl = f"（{src}{o:.1f}倍）" if o else "（オッズ未形成→上のボタンで取得）"
                    ev = st.session_state.get("exacta_ev", {}).get(f"{r['jcd']}-{r['rno']}")
                    if ev is not None:
                        evtag = (f"🟢 **2連単 買い（EV {ev:.2f}>2.0）**" if ev >= predict.EXACTA_EV_THRESHOLD
                                 else f"⚪ 2連単 見送り（EV {ev:.2f}<2.0）")
                    else:
                        evtag = "（2連単EV: 締切間際に上のEVボタン）"
                    c2.markdown(f"**単勝 {r['honmei']}号**{odlbl} {nm} ／ "
                                f"2連単 上位3点 {combos}  \n{evtag}　⏳ 結果待ち")
                    pex = r.get("power_ex")
                    if pex:
                        c2.markdown(f"💪 実力筋2連単 {' ・ '.join(pex)}　"
                                    f"<span style='color:gray'>(本命⇄対抗={r.get('taiko')}号・"
                                    f"位置that崩れた時の実力ワンツー)</span>",
                                    unsafe_allow_html=True)
                    if trio:
                        c2.markdown(f"🎲 3連複4点 {trio}　"
                                    f"<span style='color:gray'>(荒れ読みの頑健な器・別枠)</span>",
                                    unsafe_allow_html=True)
                    if tfecta:
                        c2.markdown(f"🎰 3連単3点 {tfecta}　"
                                    f"<span style='color:gray'>(大きい方・fat-tailで脆い・別枠)</span>",
                                    unsafe_allow_html=True)


def page_record():
    st.header("📊 成績（予想 vs 実際）")
    ledger = papertrade._load()
    settled = sorted([r for r in ledger if r.get("settled")],
                     key=lambda r: (r["date"], r["jcd"], r["rno"]))
    if not settled:
        st.info("精算済みの記録がまだありません（毎朝9時に自動精算されます）。")
        return

    ps = papertrade.portfolio_stats(settled)
    # --- 主表示: 単勝1.5倍未満のレースを丸ごと見送った成績 ---
    FLOOR = papertrade.ODDS_FLOOR
    kept = [r for r in settled if (r.get("final_odds") or 0) >= FLOOR]
    skipped = len(settled) - len(kept)
    f_tan_st = len(kept) * 100
    f_tan_ret = sum(r.get("tansho_return", 0) for r in kept)
    f_ex_st = sum(r.get("exacta_points", 0) for r in kept) * 100
    f_ex_ret = sum(r.get("exacta_return", 0) for r in kept)
    f_tot_st, f_tot_ret = f_tan_st + f_ex_st, f_tan_ret + f_ex_ret

    def roi(ret, st_): return f"{ret/st_:.1%}" if st_ else "―"
    st.caption(f"**単勝1.5倍未満のレースは丸ごと見送った成績**（買った {len(kept)} / 見送り {skipped} レース）")
    c = st.columns(4)
    c[0].metric("買ったレース", len(kept))
    c[1].metric("単勝 回収率", roi(f_tan_ret, f_tan_st), f"{f_tan_ret - f_tan_st:+,}円")
    c[2].metric("2連単 回収率", roi(f_ex_ret, f_ex_st), f"{f_ex_ret - f_ex_st:+,}円")
    c[3].metric("合計 回収率", roi(f_tot_ret, f_tot_st), f"{f_tot_ret - f_tot_st:+,}円")

    render_sensor(expanded=True)   # 持続センサー（本日ページと共通）

    # 💎最強妙味の実績（backtest基準 vs 前向き）
    sg = ps.get("strongest", {})
    with st.expander("💎 最強妙味（高確信フラグ）の記録（本命≠1号×本命勝率≥45%×締切2連単EV>3.5）", expanded=True):
        st.caption("かつて『唯一の勝ち筋』とした最濃断面だが、その592%はsurvivorship水増し"
                   "（1号飛びレースを結果から拾ってた幻）で撤回済。今は儲け表示でなく、条件が全部揃った"
                   "『高確信フラグ』の表示用（ROI主張はしない）。")
        c = st.columns(2)
        c[0].metric("最強妙味 592%", "撤回済", "survivorship水増し・6艇クリーンで消滅")
        if sg.get("races"):
            roi_s = sg["ret"] / sg["stake"] if sg["stake"] else 0
            c[1].metric(f"前向き実績（{sg['races']}レース）", f"{roi_s*100:.0f}%",
                        f"{sg['pl']:+,}円 / 的中{sg['hit']}")
        else:
            c[1].metric("前向き実績", "N=0", "まだ発生せず（EV捕捉が回り次第）")
        st.caption("※前向きは締切EV捕捉が要る＝ev-capture稼働後から。ここは高確信フラグの前向き記録で、"
                   "6艇クリーンでも回収100%割れ（市場効率）に収束するかを実測する装置。")

    # 参考: 無フィルタ土台(track A=机上188.7の直接対照)
    with st.expander("参考：無フィルタ土台（イン崩れ予想の全レース記録）"):
        st.caption(f"全 {ps['races']} レース記録。単勝 {roi(ps['tansho']['ret'], ps['tansho']['stake'])}"
                   f" / 2連単 {roi(ps['exacta']['ret'], ps['exacta']['stake'])}"
                   f" / 合計 {roi(ps['total']['ret'], ps['total']['stake'])}。"
                   "台帳は何を張ったかに関わらず全レース記録。かつて机上116/188.7と誇ったがsurvivorship水増しで撤回、"
                   "6艇クリーンでは単勝95%/2連単82%＝100%割れ・市場効率。上の主表示は単勝<1.5を見送った実運用版。")

    # 合計(単勝+2連単)の累積損益（フィルタ版=買ったレースのみ）
    acc, cum = 0, []
    for r in kept:
        acc += ((r.get("tansho_return", 0) - 100)
                + (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100))
        cum.append(acc)
    if cum:
        st.line_chart(pd.DataFrame({"累積損益(単勝<1.5見送り・円)": cum}))

    st.markdown("#### 明細（1点100円・単勝1点＋2連単3点）")
    tbl = []
    for r in reversed(settled):
        bet_race = (r.get("final_odds") or 0) >= FLOOR   # 単勝1.5未満は見送り＝損益0
        e_stake = r.get("exacta_points", 0) * 100
        t_pl = (r.get("tansho_return", 0) - 100) if bet_race else 0
        e_pl = (r.get("exacta_return", 0) - e_stake) if bet_race else 0
        od = f"{r.get('final_odds'):.1f}倍" if r.get("final_odds") else "―"
        tbl.append({
            "日付": r["date"],
            "レース": f"{r['venue']}{r['rno']}R",
            "対象": "買" if bet_race else "見送り",
            "単勝": f"{r['honmei']}号({od})",
            "単勝結果": (f"🎯 +{r.get('tansho_return', 0) - 100}円"
                        if r.get("tansho_win") else "× -100円") if bet_race else "―(見送り)",
            "2連単 買い目": " ".join(r.get("exacta3", [])),
            "2連単 結果": (f"🎯 {r.get('exacta_result')} 払戻{r.get('exacta_return')}円"
                          if r.get("exacta_win")
                          else f"× 結果{r.get('exacta_result') or '?'} -{e_stake}円") if bet_race else "―(見送り)",
            "単勝損益": t_pl,
            "2連単損益": e_pl,
            "レース収支": t_pl + e_pl,
        })
    st.dataframe(
        pd.DataFrame(tbl), hide_index=True, use_container_width=True,
        column_config={
            "単勝損益": st.column_config.NumberColumn(format="%+d"),
            "2連単損益": st.column_config.NumberColumn(format="%+d"),
            "レース収支": st.column_config.NumberColumn(format="%+d"),
        })
    st.caption("※単勝1点＋2連単3点/レース・各100円の実トラック（無フィルタ土台の前向き記録）。"
               "かつての2連単188.7%はsurvivorship水増しで撤回、6艇クリーンでは82%＝当たっても控除ぶん負ける。"
               "**判断は年単位**（イン崩れは1日0〜2本・高配当帯的中は年に数十本）＝数十本で結論を出さない。"
               "中間集計での一喜一憂が最大の敵。見るなら四半期に一度、累積ROIとサンプル数で。")


def render_prediction(rows):
    recs = []
    for e in rows:
        recs.append({
            "印": e["mark"], "艇": e["lane"], "選手": e.get("name") or "―",
            "級": e.get("racer_class", ""),
            "予想勝率%": e["win_pct"],
            "単勝オッズ": e.get("odds"),
            "期待値EV": e.get("ev"),
            "展示T": e.get("tenji_time"),
            "全国勝率": e.get("nat_win"),
        })
    df = pd.DataFrame(recs)

    def hl(row):
        c = LANE_COLOR.get(row["艇"], "#fff")
        txt = "#fff" if row["艇"] in (2, 3, 4, 6) else "#333"
        return [f"background-color:{c};color:{txt}" if col == "艇" else "" for col in row.index]

    sty = (df.style.apply(hl, axis=1)
           .format({"予想勝率%": "{:.1f}", "単勝オッズ": "{:.1f}", "期待値EV": "{:.2f}",
                    "展示T": "{:.2f}", "全国勝率": "{:.2f}"}, na_rep="―"))
    st.dataframe(sty, hide_index=True, use_container_width=True)


def page_race():
    st.header("レース個別予想")
    col = st.columns(4)
    hd = col[0].text_input("日付(YYYYMMDD)", dt.date.today().strftime("%Y%m%d"))
    jname = col[1].selectbox("場", [f"{c}:{n}" for c, n in VENUES.items()],
                             index=list(VENUES).index(LOCAL_VENUES[0]))
    jcd = jname.split(":")[0]
    rno = col[2].selectbox("レース", list(range(1, 13)))
    col[3].write(""); go = col[3].button("予想する", type="primary", use_container_width=True)

    if not go:
        return
    bundle = get_model()
    if bundle is None:
        st.error("モデル未学習です。`python -m src.train` を先に。")
        return
    with st.spinner(f"{venue_name(jcd)} {rno}R を予測中…（出走表・直前情報・オッズ取得）"):
        try:
            rows = predict.predict_race(hd, jcd, int(rno), bundle)
        except RuntimeError as e:
            st.error(str(e)); return
    if not rows:
        st.warning("このレースの出走表が取得できません（非開催 or 未発表）。")
        return

    rec = predict.recommend(rows)
    if rec.get("bet"):
        st.success(
            f"⚑ **イン崩れ予想（本命≠1号）**　モデルがイン(1号艇)を否定 → "
            f"**単勝 {rec['tansho']}号 {rec.get('tansho_name','')}** ／ "
            f"2連単 上位3点 **{' ・ '.join(rec['exacta3'])}**", icon="🔥")
    else:
        st.caption(f"― 見送り推奨。{rec.get('reason','')}")

    render_prediction(rows)
    st.caption("◎本命 ○対抗 ▲単穴 △連下 ｜ EV=校正済み勝率×単勝オッズ（>1.0で割安）｜ "
               "予想勝率はレース内正規化(合計100%)")


def page_about():
    st.header("📖 解説 ― 何をしていて、どこまで信じてよいか")
    n = len([r for r in papertrade._load() if r.get("settled")])

    st.markdown("""
#### 何をしているか
競艇は「1号艇（イン）」が構造的に有利で、公衆はインを過剰に人気にする傾向がある。
このサイトは公式データ（B番組表／K競走成績）で学習した機械学習モデルで各レースの1着確率を出し、
**モデルの本命が1号艇「以外」になったレース＝『イン崩れ予想（本命≠1号）』**を出す。
「公衆＝インに寄せる／モデル＝インを嫌う」がズレた場所だが、2026-07に検証をやり直すと**そこに儲けのエッジは無い**（下記）。

買い方は各レース **単勝（本命1点）＋2連単（確率上位3点）・各100円**。
さらに二段でフィルタ：**①単勝<1.5倍のレースは丸ごと見送り**（大本命＝荒れず単勝も2連単も控除率負け）、
**②2連単は締切間際のEV（＝確率×締切オッズ）が2.0超のときだけ買う**（後述）。
""")

    st.markdown("#### 検証で分かったこと（2026-07 やり直し）")
    c = st.columns(3)
    c[0].metric("イン崩れ 単勝", "95%", "6艇クリーン・確定払戻・3年")
    c[1].metric("イン崩れ 2連単3点", "82%", "6艇クリーン・100%割れ")
    c[2].metric("イン崩れ 3連単3点", "92%", "6艇クリーンでもエッジ無し")
    st.markdown("""
上は**6艇そろい（全6艇が枠に残った＝ベット時に再現できる状態）**だけで、確定払戻ベース・3年ウォークフォワードで測った正直な数字。
**全部100%割れ＝競艇も市場効率＝エッジ無し**（当たっても控除ぶん負ける）。

かつてこのページは単勝116%/2連単188%と誇っていたが、あれは**survivorship水増しの幻**だった：
学習・検証の枠が `win IS NOT NULL` で絞られ、事故・失格・欠場（win=NULL）の艇が黙って消えていた。
1号（イン）が飛んだレースは1号が結果から消え、モデルは自動で非イン艇を本命にさせられ「妙味（本命≠1号）」と後付けラベルされる
＝**高配当の荒れレースを結果を見てから拾っていた**だけ。ベット時には6艇そろっているので再現不能。だから撤回した。
""")

    st.markdown("#### まだ分かっていないこと（正直に）")
    _live = "統計的に無意味" if n < 100 else "蓄積途上"
    st.markdown(f"""
- **前向きの実記録でも回収100%割れが続くかを確認中**（毎朝自動・現在 {n} レース）。今は{_live}で、
  はっきり言えるのは**年単位**（下記）。予想は**エッジ無し前提**なので、100%割れ(🔴)が想定どおりの正解。
- **かつて『高配当に寄ると儲かる（割安拾い）』と考えていたのは撤回**。あれはsurvivorship水増しの副産物で、
  6艇クリーンで測り直すと消える。予測構造（#1=イン/位置・#2=地力/実力・+展開/相対ST）は本物だが、
  公開情報ゆえ市場が先に織り込み済で儲けにはならない。
- **予測が特定期間に偏っていないか**は、別の期間データが増えないと詰めきれない（結論=エッジ無しは動きにくいが）。
""")

    st.markdown("#### 年間どれくらいのレースが対象か／『年単位』の意味")
    st.table(pd.DataFrame({
        "年": ["2023", "2024", "2025"],
        "総レース": ["55,272", "55,240", "55,066"],
        "イン崩れ予想(本命≠1号)": ["1,254", "1,225", "1,183"],
        "出現率": ["2.3%", "2.2%", "2.1%"],
    }))
    st.markdown("""
**イン崩れ予想は年 約1,200本**（全レースの約2.2%・1日ならし約3本、0本の日も普通）。単勝<1.5倍を丸ごと見送ると年 約1,000本。

本数は少なくないが、**賭式で分散がまるで違う**のが肝：
- **単勝は低分散** → 年1,200本あれば **約1年** で回収率（＝市場効率で100%割れ）が読めてくる
- **2連単は稀な高配当が牽引するファットテール** → 高配当帯は**年たった約84レース**しかなく、
  CIが広いまま。**回収が本当に100%割れか腰を据えて見るのは2〜3年**

だから「放っておいて積む」のが正解。中間集計の一喜一憂（今日勝った/負けた）は最大の敵で、見るなら四半期に一度。
""")

    st.markdown("#### 記録は3本立て（混同しないため）")
    st.markdown("""
- **track A＝無フィルタ土台**：イン崩れ予想を全部記録した台帳（「成績」ページ）
- **reco＝フィルタ版（参考）**：単勝<1.5を見送った版。track Aから導く目安
- **実弾＝実際に張った分**：裁量・金額込みの別track

いずれも回収は6艇クリーンの単勝95%/2連単82%（＝100%割れ）に収束するはず＝儲けの証明用ではなく、市場効率の前向き確認用。
""")

    st.markdown("#### 最新の状況（2026-07 更新）")
    st.markdown("""
- **2連単EVフィルタは撤回扱い**：締切間際のオッズで EV=確率×オッズ を出す仕組みは残すが、
  かつて「この帯は堅い（512% / 305% / CI[435,597]）」としたのはsurvivorship水増しの幻で、6艇クリーンでは消える。
  今は「⚑イン崩れ予想」の🎯EVボタン→EV>2.0が🟢表示になるが、**参考・儲け主張なし**。
- **成績ページを「単勝<1.5見送り版」に**：実際に張る想定（チャラレース抜き）の回収率を主表示。無フィルタ土台は参考に格納。
- **特徴量を掃除（24→14）**：ROI-permutationで「効いてない軸」を三重確認して除去（回収率は不変）。
  分かったこと＝**当てる本体は特徴でなく『イン崩れを選ぶこと（selection）』**。ただしそれも儲けには化けない（市場効率）。
- **自動化をクラウド（GitHub Actions）へ移設＝PC不要**：毎朝9時に自動でscan→結果反映→サイト更新。PCを切っても回る。
- **ライブ生死ラインを事前固定（プレレジ）**：数字を見てから基準を決めると人は都合よく解釈するので、
  「何本たまって回収△%を割ったら畳む／超えたら信じる」を先に固定済み（docs/LIVE_PREREG.md）。
  特にEV>2.0は**約300本(1年+)たまるまで良し悪しを判断しない**。
""")

    st.markdown("#### 使い方・免責")
    st.info(
        "半自動です（自動購入はしません）。「⚑イン崩れ予想」の買い目をテレボート等へ**自分で入力**して投票。"
        "本サイトはシステム構築の研究目的で、実弾は少額（100円）想定。"
        "**統計的推定であり的中・利益を保証しません。投資は自己責任で。**"
    )


AGENT_UNIT = 1000            # エージェントの1点あたり張額（円）
AGENT_EV_TH = 2.0


def _agent_pl(r, sc):
    """エージェントの1レースP&L（円）。単勝<1.5は見送り(None)。2連単はEV>2.0のみ。"""
    if (r.get("final_odds") or 0) < papertrade.ODDS_FLOOR:
        return None
    pl = (r.get("tansho_return", 0) - 100) * sc
    bet2 = (r.get("exacta_ev") or 0) >= AGENT_EV_TH
    if bet2:
        pl += (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100) * sc
    return pl


def page_pro():
    st.header("🎯 プロ・エージェント（毎日の予想と結果）")
    st.caption(f"逐次エージェント＝三関門（本命≠1号艇 → 単勝≥1.5 → 2連単EV>2.0）で選び、"
               f"1点 {AGENT_UNIT:,}円で張る。毎日ここに予想と結果が積まれる。"
               f"※かつて『机上+25万/月』と出していたがsurvivorship水増しで撤回。6艇クリーンでは回収100%割れ＝負け寄り。負け月も隠さず出す。")
    sc = AGENT_UNIT / 100
    ledger = papertrade._load()
    today = dt.date.today().strftime("%Y%m%d")

    # --- 今日の予想 ---
    st.markdown("#### 📅 今日の予想")
    todays = [r for r in ledger if r.get("date") == today and not r.get("settled")]
    if not todays:
        c = scan.load_cache()
        if c and c.get("date") == today and len(c.get("picks", [])) == 0:
            st.success("本日はイン崩れ予想無し＝**全レース見送り**（プロは買わない日）。")
        else:
            st.info("本日の対象レースはまだありません（毎朝9時に自動更新）。")
    else:
        for r in todays:
            ev = r.get("exacta_ev")
            ev_txt = (f"EV **{ev:.2f}** → {'🟢買い' if ev >= AGENT_EV_TH else '⚪見送り'}"
                      if ev is not None else "EV：締切間際に判定")
            combos = " ・ ".join(r.get("exacta3", []))
            st.markdown(
                f"**{r['venue']}{r['rno']}R**（締切 {r.get('deadline','—')}）  \n"
                f"　🎫 **単勝：{r['honmei']}号**（各1,000円・締切で1.5倍以上か判定）  \n"
                f"　🎫 **2連単：{combos}**（各1,000円・{ev_txt}）")

    # --- 確定した結果（日別） ---
    settled = sorted([r for r in ledger if r.get("settled")],
                     key=lambda r: (r["date"], r["jcd"], r["rno"]))
    st.markdown("#### 📊 確定した結果（エージェントの実績）")
    if not settled:
        st.caption("まだ確定結果がありません。毎日積まれます。")
    else:
        by_day, run = {}, 0
        for r in settled:
            pl = _agent_pl(r, sc)
            if pl is None:
                continue
            d = by_day.setdefault(r["date"], {"pl": 0.0, "n": 0})
            d["pl"] += pl; d["n"] += 1
        rows = []
        for dt_ in sorted(by_day):
            run += by_day[dt_]["pl"]
            rows.append({"日付": f"{dt_[4:6]}/{dt_[6:]}", "賭けたレース": by_day[dt_]["n"],
                         "その日の収支": int(by_day[dt_]["pl"]), "累計": int(run)})
        if rows:
            tot_n = sum(d["n"] for d in by_day.values())
            tot_pl = int(run)
            cc = st.columns(3)
            cc[0].metric("賭けたレース累計", tot_n)
            cc[1].metric("累計収支", f"{tot_pl:+,}円")
            cc[2].metric("1点あたり", f"{AGENT_UNIT:,}円")
            st.dataframe(pd.DataFrame(reversed(rows)), hide_index=True, use_container_width=True,
                         column_config={"その日の収支": st.column_config.NumberColumn(format="%+d"),
                                        "累計": st.column_config.NumberColumn(format="%+d")})
            # 月別
            by_mo = {}
            for dt_ in by_day:
                by_mo.setdefault(dt_[:6], 0.0)
                by_mo[dt_[:6]] += by_day[dt_]["pl"]
            st.markdown("###### 月別")
            st.dataframe(pd.DataFrame([{"月": f"{m[:4]}/{m[4:]}", "収支": int(v)}
                                       for m, v in sorted(by_mo.items())]),
                         hide_index=True, use_container_width=True,
                         column_config={"収支": st.column_config.NumberColumn(format="%+d")})
            # 賭けたレースごとの明細（買い目＋結果）
            det = []
            for r in settled:
                if _agent_pl(r, sc) is None:
                    continue
                bet2 = (r.get("exacta_ev") or 0) >= AGENT_EV_TH
                od = f"{r.get('final_odds'):.1f}倍" if r.get("final_odds") else "―"
                det.append({
                    "日付": f"{r['date'][4:6]}/{r['date'][6:]}",
                    "レース": f"{r['venue']}{r['rno']}R",
                    "単勝 買い目": f"{r['honmei']}号({od})",
                    "単勝 結果": "🎯的中" if r.get("tansho_win") else "×ハズレ",
                    "2連単 買い目": " ".join(r.get("exacta3", [])) if bet2 else "見送り(EV<2.0)",
                    "2連単 結果": (f"🎯{r.get('exacta_result')} 払戻{r.get('exacta_return')}円"
                                  if bet2 and r.get("exacta_win")
                                  else ("×" + str(r.get("exacta_result") or "") if bet2 else "―")),
                    "レース収支": int(_agent_pl(r, sc)),
                })
            st.markdown("###### 明細（賭けたレースごとの買い目と結果）")
            st.dataframe(pd.DataFrame(list(reversed(det))), hide_index=True, use_container_width=True,
                         column_config={"レース収支": st.column_config.NumberColumn(format="%+d")})
        else:
            st.caption("エージェントが賭けた確定レースはまだありません（単勝<1.5は見送り／2連単はEV>2.0のみ）。")

    st.info("※これは前向きの実記録＝ライブ検証。かつての机上+25万/月はsurvivorship水増しで撤回、"
            "回収が本当に100%割れ（市場効率）かを積んで確かめる装置。"
            "数十本〜年単位で見る（プレレジ：EV枠はN<300本は判断しない）。")

    # --- 暗黙知ノート（頭の中） ---
    st.markdown("---")
    st.markdown("#### 🧠 エージェントの頭の中（データに無い暗黙知）")
    path = os.path.join(os.path.dirname(__file__), "data", "pro_wisdom.md")
    if os.path.exists(path):
        with st.expander("開く：気配と地合いの読み方"):
            st.markdown(open(path, encoding="utf-8").read())


def page_wild():
    st.header("🌊 本日のインが崩れやすいレース（参考）")
    st.caption("モデルのイン勝率that低い＝1号が信頼薄い順（本命that1号でも勝率が薄ければ入る）。"
               "**万舟が来る/儲かる保証ではない**（検証: 万舟率は条件によらず約24%で一定）。"
               "捉えてるのは『インの脆さ』だけ＝見て楽しむ/参考用。")
    ls = wild.log_stats()
    if ls:
        st.success(f"📊 **検証（ほんとに荒れたか）**：荒れ予想 **{ls['n']}レース** ／ "
                   f"本当に荒れた（1号飛び） **{ls['broke']*100:.0f}%** ／ 万舟(3連単≥1万) {ls['manshu']*100:.0f}% ／ "
                   f"平均3連単配当 {ls['avg_yen']:,.0f}円")
        st.caption("※全レースの1号飛び率は約46%。フラグ群がこれを上回れば『インの脆さ』を当ててる。"
                   "万舟率≈24%は条件で動かない（荒れは当てても万舟は事前に読めない、の裏取り）。")
        # この荒れ予想を買えたのか＝本命単勝の勝敗（全フラグ／本命≠1号）
        ta, tc = ls.get("tan_all"), ls.get("tan_contra")
        if ta:
            line = (f"💰 **買えたのか（モデル本命の単勝1点）**：全フラグ {ta['n']}件 "
                    f"的中{ta['hit']*100:.0f}% 回収{ta['roi']*100:.0f}% 収支{ta['pl']:+,}円")
            if tc:
                line += (f"　／　**本命≠1号だけ（＝実際に張るイン崩れ）{tc['n']}件 "
                         f"的中{tc['hit']*100:.0f}% 回収{tc['roi']*100:.0f}% 収支{tc['pl']:+,}円**")
            st.markdown(line)
            st.caption("※荒れ予想には本命=1号(弱いイン)も混じる＝それを本命単勝で買うのはイン賭け＝分が悪い。"
                       "実際に張るのは本命≠1号のサブセット。確定払戻・少額サンプルはノイズ。")
    w = wild.load()
    today = dt.date.today().strftime("%Y%m%d")
    if not w or not w.get("races"):
        st.info("まだ荒れ度データがありません（毎朝の自動スキャンで生成されます）。")
        return
    if w.get("date") != today:
        st.caption(f"（直近スキャン: {w['date'][4:6]}/{w['date'][6:]} 分）")
    for i, r in enumerate(w["races"], 1):
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {i}. {r['venue']} {r['rno']}R")
            if r.get("deadline"):
                c1.caption(f"⏰ 締切 {r['deadline']}")
            c1.metric("荒れ度", f"{r['score']:.0f}")
            reasons = " ／ ".join(r.get("reasons") or []) or "（際立った条件なし）"
            c2.markdown(f"**荒れ要因**：{reasons}")
            c2.caption(f"モデル本命 {r['honmei']}号（勝率 {r.get('win_pct', 0):.0f}%）"
                       + ("　＝インを疑ってる" if r.get("honmei") != 1 else "　＝インだが薄い"))
            res = _live_result(w["date"], r["jcd"], r["rno"])
            if res and res.get("winner") is not None:
                broke = res["winner"] != 1
                tf = res.get("trifecta_yen")
                man = " ／💥万舟" if (tf and tf >= 10000) else ""
                tag = "🔥 荒れた！（1号飛び）" if broke else "⚪ 不発（1号残り）"
                paytxt = f" ／ 3連単 {tf:,}円{man}" if tf else ""
                c2.markdown(f"**結果：{tag}**（勝ち {res['winner']}号{paytxt}）")
            else:
                c2.caption("⏳ 結果待ち")
    st.caption("※スコアは 1-モデルのイン勝率 / 1号の級 / 混戦度 / アウトの実力 の合成。"
               "インの脆さは捉えるが、それが小穴で終わるか万舟に化けるかは事前に読めない(=買い推奨ではない)。")


@st.fragment(run_every=180)
def _agent_feed(log):
    races, date = log["races"], log["date"]
    bets = [r for r in races if r.get("bet")]
    mikoku = [r for r in races if not r.get("bet")]
    pl = done = hitraces = 0
    feed = []
    for r in bets:
        res = _live_result(date, r["jcd"], r["rno"])
        feed.append((r, res))
        if res:
            done += 1
            ret = (res.get("tansho_yen") or 0) if res.get("winner") == r["honmei"] else 0
            ret += (res.get("exacta_yen") or 0) if res.get("exacta_combo") in (r.get("exacta3") or []) else 0
            ret += (res.get("trio_yen") or 0) if res.get("trio_combo") in (r.get("trio4") or []) else 0
            ret += (res.get("trifecta_yen") or 0) if res.get("trifecta_combo") in (r.get("trifecta3") or []) else 0
            pl += ret - 11 * 100          # 単1+2連3+3複4+3単3=11点
            if ret > 0:
                hitraces += 1
    c = st.columns(4)
    c[0].metric("検討レース", len(races))
    c[1].metric("勝負", len(bets))
    c[2].metric("結果確定", f"{done}/{len(bets)}")
    c[3].metric("本日収支(11点)", f"{pl:+,}円")
    st.caption(f"的中 {hitraces} レース。1レース=単勝1＋2連単3＋3連複4＋3連単3＝11点(1,100円)。"
               "結果はレース後およそ5分で自動反映（3分毎更新）。")
    st.divider()
    for r, res in feed:
        with st.container(border=True):
            st.markdown(f"**{r.get('deadline','')}　{r['venue']}{r['rno']}R**　"
                        + commentary.comment(r, res))
    if mikoku:
        with st.expander(f"⚪ 見送り {len(mikoku)}レース（インが濃厚＝イン崩れ無しと判断した回）"):
            for r in mikoku:
                st.caption(f"{r.get('deadline','')} {r['venue']}{r['rno']}R ｜ "
                           f"本命{r['honmei']}号({r.get('win_pct',0):.0f}%) 見送り")


def page_agent():
    st.header("🎙️ エージェント実況（全レース予想→勝負/見送り→結果）")
    st.caption("エージェントが本日の全レースを直前まで読み、イン崩れ(本命≠1号)だけ勝負。結果が出たら一言。自動更新(3分毎)。"
               "※コメントはルール生成(LLM不使用)・儲け保証なし（6艇クリーンで回収100%割れ＝市場効率）。実弾は自己責任。")
    log = commentary.load_log()
    today = dt.date.today().strftime("%Y%m%d")
    if not log or not log.get("races"):
        st.info("まだ本日のエージェントログがありません（毎朝の自動スキャンで生成されます）。")
        return
    if log.get("date") != today:
        st.caption(f"（直近スキャン: {log['date'][4:6]}/{log['date'][6:]} 分）")
    _agent_feed(log)


@st.cache_data(ttl=3600, show_spinner=False)
def _trifecta_ranking(date, jcd, rno):
    """レースのモデル3連単ランキング(120通り・確率降順)。1時間キャッシュ。"""
    rows = predict.predict_race(date, jcd, rno, get_model())
    if not rows:
        return None
    wp = {e["lane"]: e["win_prob"] for e in rows}
    return [c for c, _ in sorted(predict.harville_trifecta(wp).items(), key=lambda x: -x[1])]


def page_trifecta():
    st.header("🎰 3連単ページ（予想／20点買い／何点で当たるか）")
    st.caption("モデルの3連単予想と、20点買った場合の結果、そして『実際は何点目で当たったか』"
               "＝当たりに必要な点数。3連単は着順あり120通り、20点＝上位20。確定払戻・儲け保証なし。")
    today = dt.date.today().strftime("%Y%m%d")
    cache = scan.load_cache()
    bets = (cache or {}).get("picks", []) if cache and cache.get("date") == today else []

    st.subheader("本日のイン崩れ予想")
    if not bets:
        st.caption("本日のイン崩れ予想なし、またはスキャン待ち。")
    for p in bets:
        with st.container(border=True):
            st.markdown(f"### {p['venue']}{p['rno']}R　締切{p.get('deadline','')}")
            with st.spinner("モデル3連単予想を計算中…"):
                ranking = _trifecta_ranking(today, p["jcd"], p["rno"])
            if not ranking:
                st.caption("予測不可（出走表未形成）")
                continue
            st.markdown("**モデル予想（買い目・上位20点）**")
            st.markdown("　".join(f"{i+1}.{c}" for i, c in enumerate(ranking[:20])))
            res = _live_result(today, p["jcd"], p["rno"])
            if not res or not res.get("trifecta_combo"):
                st.caption("⏳ 結果待ち（レース後およそ5分）")
                continue
            win, yen = res["trifecta_combo"], res.get("trifecta_yen") or 0
            rank = ranking.index(win) + 1 if win in ranking else None
            st.markdown(f"**実際：{win}（{yen:,}円）**")
            if rank:
                st.markdown(f"🎯 **必要点数 {rank}点**（上位から{rank}点買えば的中）")
                if rank <= 20:
                    st.success(f"20点内で的中（予想{rank}番目）！{yen:,}円　賭2,000円 → 収支 {yen-2000:+,}円")
                else:
                    st.warning(f"20点では届かず。当たるまで {rank}点必要（20点＝−2,000円）")
                with st.expander(f"上位から当たるまでの買い目 全{rank}点を見る"):
                    st.markdown("　".join(f"{i+1}.{c}" for i, c in enumerate(ranking[:rank])))

    st.subheader("これまでの記録：何点買えば当たってたか")
    st.caption("前向き台帳の3連単を、実結果that予想の何位か(必要点数)で集計。分布that『何点買えば当たるか』の答え。")
    settled = [x for x in papertrade._load()
               if x.get("settled") and x.get("trifecta_rank")
               and (x.get("res_full") or {}).get("trifecta_combo")]
    ranks = []
    for r in settled:
        win = r["res_full"]["trifecta_combo"]
        rk = r["trifecta_rank"]
        if win in rk:
            ranks.append(rk.index(win) + 1)
    if not ranks:
        st.caption("まだ記録なし（明日以降の全ランク記録付き settle から溜まります）。")
        return
    import numpy as np
    st.markdown(f"記録 **{len(ranks)}レース**｜必要点数 中央値 **{int(np.median(ranks))}点** ／ 平均 {np.mean(ranks):.0f}点")
    for k in [1, 3, 6, 12, 20, 40, 60, 120]:
        hit = sum(1 for x in ranks if x <= k) / len(ranks)
        st.markdown(f"- 上位 **{k}点** 買い → 的中率 {hit*100:.0f}%")


def page_dynamic():
    st.header("🎯 動的買い目（EV割安拾い＝毎レース点数が変わる）")
    st.caption("モデル確率×締切オッズでEVを出し、EV>閾値の『割安な目』だけ買う。割安が多い→点数増、"
               "少ない→絞る＝『20倍つきそうなら20点』の正体（見た目の荒れでなくEV）。2連単EV>2.0は検証済／"
               "3連系は実験（3連単ライブ収集で調整中）。締切間際に押すのが正確。儲け保証なし。")
    th = st.slider("EV割安ライン（これ超えを買う）", 1.5, 4.0, 2.0, 0.1)
    st.caption("📈 かつて『backtestでEV>2.0=316% / 2.5=344% / 3.0=370%』としたが、これはsurvivorship水増しで撤回。"
               "6艇クリーンでは2連単82%＝100%割れ（市場効率）。EVで割安を拾う仕組みは残すが儲け主張なし・参考表示。"
               "ライブ閾値は2.0で凍結（再最適化＝過学習を避ける）。上げるほど点数は減る。")
    today = dt.date.today().strftime("%Y%m%d")
    cache = scan.load_cache()
    bets = (cache or {}).get("picks", []) if cache and cache.get("date") == today else []
    if not bets:
        st.info("本日のイン崩れ予想なし、またはスキャン待ち。")
        return
    names = {"tansho": "単勝", "exacta": "2連単", "trio": "3連複", "trifecta": "3連単"}
    # 持続センサー→賭式ゲート（DECAYEDは自動OFF＝ライブ劣化を買い目に反映）
    gate = sensor.bet_gate()
    off = [g["track"] for g in gate.values() if not g["ok"]]
    if off:
        st.error("🔴 持続センサーが崩れ判定 → 自動OFF中の賭式：" + " / ".join(off))
    else:
        st.caption("🟢 持続センサー：全賭式ON（DECAYEDなし＝崩れ確定まで張る）")
    if st.button("🎯 現在の締切オッズで動的判定"):
        bundle = get_model()
        for p in bets:
            with st.container(border=True):
                st.markdown(f"### {p['venue']} {p['rno']}R　締切{p.get('deadline','')}")
                with st.spinner("全賭式の締切オッズ取得中…"):
                    r = predict.dynamic_for_race(today, p["jcd"], p["rno"], bundle, th, gate)
                if not r:
                    st.caption("予測不可（出走表/オッズ未形成）")
                    continue
                rows, d = r
                if d.get("strongest"):
                    st.success(f"💎 **最強妙味（高確信フラグ）！**　本命勝率{d['p0']*100:.0f}% × 2連単EV{d['ex_ev3']:.1f}（>3.5）"
                               "＝本命≠1号×高確信×強割安が全部揃った表示用フラグ（かつての592%はsurvivorship水増しで撤回・儲け主張なし）")
                elif d.get("ex_ev3") is not None:
                    st.caption(f"💎条件 本命勝率≥45%×2連単EV>3.5 ／ 現在 {d['p0']*100:.0f}% × EV{d['ex_ev3']:.1f}")
                if d["points"] == 0:
                    st.markdown(f"**割安ゼロ → 見送り**（EV>{th}の目なし＝市場が正しく値付け）")
                    continue
                st.markdown(f"**割安 {d['points']}個 → {d['points']}点（{d['cost']:,}円）**")
                for k in ["tansho", "exacta", "trio", "trifecta"]:
                    sel = d["buy"][k]
                    if not sel:
                        continue
                    top = " ／ ".join(f"{x['combo']}(EV{x['ev']}/{x['odds']:.0f}倍)" for x in sel[:6])
                    more = f" …他{len(sel)-6}個" if len(sel) > 6 else ""
                    st.markdown(f"- **{names[k]} {len(sel)}点**：{top}{more}")
    else:
        st.caption("↑ボタンで、今の締切オッズから各レースの割安な目を全賭式で洗い出す（点数はレースごとに変動）。")


def page_styles():
    st.header("💴 買い方くらべ（アクセル段階別トータル）")
    st.caption("点数を増やすほど『的中率↑・万舟捕捉↑』だが『回収率↓』（控除を余計に払う）＝"
               "効率(回収率)と総額(P&L)でベストが違う。"
               "⚠️ここの回収率(533%等)はsurvivorship水増しで撤回済＝6艇クリーンでは全部100%割れ・エッジ無し。点数と的中/回収の"
               "『関係の形』を見る参考用で、水準は信じない。")
    h = styles.HISTORICAL
    st.subheader("3年トータル（本体）")
    st.caption(h["universe"])
    st.dataframe(pd.DataFrame([{
        "買い方": r["name"], "点/R": r["pts"], "回収率": f"{r['roi']}%",
        "総収支": f"{r['pl']:+,}円", "的中率": f"{r['hit']}%",
        "万舟": r["manshu"], "最大払戻": f"{r['max']:,}円",
    } for r in h["rows"]]), hide_index=True, use_container_width=True)
    st.info("🔑 " + h["note"])

    st.subheader("点数×効率カーブ（3連単・イン崩れ3年）")
    st.caption("何点が一番効率（回収率）いいか。" + styles.TRIFECTA_CURVE["note"])
    c = styles.TRIFECTA_CURVE
    st.line_chart(pd.DataFrame({"回収率%": c["roi"], "的中率%": c["hit"]}, index=c["points"]))
    st.markdown("⚠️ **この533%/319%等はsurvivorship水増しで撤回済**（1号飛びレースを結果から拾った幻）。"
                "6艇クリーンでは3連単3点=92%＝100%割れ（市場効率・エッジ無し）。カーブは水準でなく"
                "**点数と的中/回収の関係の形**を見る参考用。点を増やすほど**的中率↑・回収率↓**の傾向だけは本物。")

    st.subheader("毎日の記録（前向き・実トラック）")
    st.caption("確定した日の各スタイル収支を毎日積む。上の3年トータルは机上、こっちが実際の前向き記録。"
               "ワイド系は全ランク記録that要るので明日のscan以降から。")
    dd = styles.daily([x for x in papertrade._load() if x.get("settled") and x.get("res_full")])
    if not dd:
        st.caption("まだ前向きの確定記録がありません（明日以降の scan→settle から毎日溜まります）。")
    else:
        names = [n for n, *_ in styles.STYLES]
        cum = {n: 0 for n in names}
        tbl = []
        for d in sorted(dd):
            row = {"日付": f"{d[4:6]}/{d[6:]}"}
            for n in names:
                c = dd[d][n]
                pl = c["ret"] - c["stake"]
                cum[n] += pl
                row[n.split("（")[0]] = f"{pl:+,}" if c["comp"] else "—"
            tbl.append(row)
        tbl.append({"日付": "▶累計", **{n.split("（")[0]: f"{cum[n]:+,}" for n in names}})
        st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

    st.subheader("本日の勝負レースを各スタイルで見ると")
    st.caption("※本日ぶんは数レース＝ノイズ。判断は上の3年トータルで。")
    today = dt.date.today().strftime("%Y%m%d")
    cache = scan.load_cache()
    bets = (cache or {}).get("picks", []) if cache and cache.get("date") == today else []
    if not bets:
        st.caption("本日の勝負レースなし、または未確定。")
        return
    agg = {name: {"stake": 0, "ret": 0, "any": False} for name, *_ in styles.STYLES}
    for p in bets:
        res = _live_result(today, p["jcd"], p["rno"])
        if not res:
            continue
        for s in styles.compute(p, res):
            if s["computable"]:
                agg[s["name"]]["stake"] += s["stake"]
                agg[s["name"]]["ret"] += s["ret"]
                agg[s["name"]]["any"] = True
    rows = []
    for name, *_ in styles.STYLES:
        a = agg[name]
        if a["any"]:
            rows.append({"買い方": name, "本日賭金": f"{a['stake']:,}円",
                         "本日払戻": f"{a['ret']:,}円", "本日収支": f"{a['ret']-a['stake']:+,}円"})
        else:
            rows.append({"買い方": name, "本日賭金": "—", "本日払戻": "—",
                         "本日収支": "全ランク記録待ち（次のscanから）"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


st.title("🚤 競艇予想")
status_banner()
def _katai_effective(date, p):
    """堅軸pickにライブ結果を付ける（未確定は _settled=False）。_live_result(120秒cache)流用。"""
    res = _live_result(date, p["jcd"], p["rno"])
    if not res:
        return {**p, "_settled": False}
    e = dict(p)
    e["_settled"] = True
    e["winner"] = res["winner"]
    e["tansho_win"] = res["winner"] == p["tansho"]
    e["tansho_return"] = res["tansho_yen"] if (e["tansho_win"] and res["tansho_yen"]) else 0
    e["final_odds"] = round(res["tansho_yen"] / 100, 1) if (e["tansho_win"] and res["tansho_yen"]) else None
    ex3 = p.get("exacta3", []) or []
    e["exacta_result"] = res["exacta_combo"]
    e["exacta_win"] = res["exacta_combo"] in ex3
    e["exacta_return"] = res["exacta_yen"] if (e["exacta_win"] and res["exacta_yen"]) else 0
    return e


def page_katai():
    st.header("🏆 本日の勝てる目（高的中の堅い本命）")
    st.caption("**当たる目だけ**を出すページ。逆張り妙味(逆に賭ける)でなく、"
               "『1にイン(位置)・2に地力(実力)』that揃った堅い本命。"
               "検証(確定払戻・6艇)：イン×地力1位で単勝**的中73.7%**、イン×地力1-2位で68.8%、"
               "モデル確信≥50%で63.3%。")
    st.warning("⚠️ ただし**回収は89〜94%＝当たっても控除で儲けは出ない**（市場that実力を正しく値付け済）。"
               "これは「当てて楽しむ／自信度」用で、儲けの主張はしない。今日確定した結論"
               "＝競艇も市場効率・エッジ無し、に沿った正直な出し方。")
    k = katai.load()
    today = dt.date.today().strftime("%Y%m%d")
    if not k or not k.get("picks"):
        st.info("まだ堅軸データthaありません（毎朝の自動スキャンで生成されます）。")
        return
    if k.get("date") != today:
        st.caption(f"（直近スキャン: {k['date'][4:6]}/{k['date'][6:]} 分）")
    _render_katai(k["date"], k["picks"])


def _deadline_passed(deadline, now_min, buffer=10):
    """締切(HH:MM)+buffer分that今(JST分)を過ぎたか。過ぎた分だけ結果を取りに行く。"""
    if not deadline:
        return False
    try:
        h, m = deadline.split(":")
        return now_min >= int(h) * 60 + int(m) + buffer
    except Exception:
        return False


@st.fragment(run_every=180)
def _render_katai(date, raw_picks):
    now = dt.datetime.utcnow() + dt.timedelta(hours=9)            # JST
    now_min, today = now.hour * 60 + now.minute, now.strftime("%Y%m%d")
    # 当日はレース済み(締切+10分経過)だけライブ精算。過去日は全件精算。開催前は0取得=即表示
    picks = [
        _katai_effective(date, p) if (date != today or _deadline_passed(p.get("deadline"), now_min))
        else {**p, "_settled": False}
        for p in raw_picks
    ]
    picks.sort(key=lambda x: (-int(x.get("_settled", False)), -x.get("hit_pct", 0)))
    settled = [p for p in picks if p.get("_settled")]
    n = len(settled)
    t_hit = sum(1 for p in settled if p.get("tansho_win"))
    t_ret = sum(p.get("tansho_return", 0) for p in settled)
    e_hit = sum(1 for p in settled if p.get("exacta_win"))
    e_ret = sum(p.get("exacta_return", 0) for p in settled)

    c = st.columns(4)
    c[0].metric("堅軸レース", f"{len(picks)} 件")
    c[1].metric("結果確定", f"{n} 件")
    c[2].metric("単勝 的中率", f"{t_hit/n*100:.0f}%" if n else "—",
                help="想定74%(◎鉄板)。確定分の実績。")
    c[3].metric("単勝 回収率", f"{t_ret/(n*100)*100:.0f}%" if n else "—",
                help="100円ずつ。<100%=当たるが控除で負ける(市場効率)。")
    if n:
        st.caption(f"実績（確定{n}件）：単勝 的中{t_hit}/{n}・回収{t_ret/(n*100)*100:.0f}% ／ "
                   f"2連単3点 的中{e_hit}/{n}・回収{e_ret/(n*300)*100:.0f}%　"
                   "※結果はレース後およそ5分で自動反映（3分毎更新）。")
    st.divider()

    for r in picks:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {r['venue']} {r['rno']}R")
            c1.markdown(f"**{r.get('tier','')}**")
            if r.get("deadline"):
                c1.caption(f"⏰ 締切 {r['deadline']}")
            ipr = r.get("in_power_rank")
            note = (f"イン地力{ipr}位・" if ipr else "") + f"確信{r.get('conf')}%"
            head = (f"**単勝 {r['tansho']}号 {r.get('name','')}**（{note}）　"
                    f"🎯想定的中 約{r.get('hit_pct')}%")
            ex3 = ' ・ '.join(r.get("exacta3") or [])
            if r.get("_settled"):
                t = (f"🎯的中 **{r.get('final_odds')}倍**" if r.get("tansho_win")
                     else f"×ハズレ（勝ち **{r.get('winner')}号**）")
                e = (f"🎯的中 {r.get('exacta_return')}円" if r.get("exacta_win")
                     else f"×ハズレ（結果 {r.get('exacta_result') or '?'}）")
                c2.markdown(f"{head}  \n"
                            f"単勝: {t} ｜ 2連単3点 {ex3}: {e}")
            else:
                c2.markdown(f"{head}  \n"
                            f"2連単 本命流し3点：{ex3}　⏳ 結果待ち")
    st.caption("※想定的中は過去3年の同型(確定払戻)の実測値。回収<100%は不変＝儲け保証なし。"
               "確定分の実績（上の指標）that想定に近いか＝前向きの答え合わせ。")


page = st.sidebar.radio("ページ",
                        ["🔥渾身の一点", "🏆勝てる目（堅軸）", "⚑イン崩れ予想（本命≠1号）", "🎯動的買い目",
                         "🎰3連単", "🎙️エージェント実況", "💴買い方くらべ", "🌊荒れそうレース",
                         "成績（予想vs実際）", "プロの見方", "レース個別予想", "解説"])
st.sidebar.markdown("---")
st.sidebar.caption("愛知近郊: " + " / ".join(venue_name(j) for j in LOCAL_VENUES))
if page == "🔥渾身の一点":
    page_ichiten()
elif page == "⚑イン崩れ予想（本命≠1号）":
    page_today()
elif page == "🏆勝てる目（堅軸）":
    page_katai()
elif page == "🎯動的買い目":
    page_dynamic()
elif page == "🎰3連単":
    page_trifecta()
elif page == "🎙️エージェント実況":
    page_agent()
elif page == "💴買い方くらべ":
    page_styles()
elif page == "🌊荒れそうレース":
    page_wild()
elif page == "成績（予想vs実際）":
    page_record()
elif page == "プロの見方":
    page_pro()
elif page == "解説":
    page_about()
else:
    page_race()
