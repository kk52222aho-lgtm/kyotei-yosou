"""競艇予想サイト（Streamlit 版）。

  streamlit run streamlit_app.py

既存の src モジュール（モデル・予測・スキャン）をそのまま利用する。
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import streamlit as st

from src import predict, papertrade, scan, scraper, sensor, wild, commentary
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
        "✅ **机上の前進検証クリア**: 戦略未使用の3年(2023-25)で 単勝 平均116% / 2連単3点 ~188%"
        "（計3,662ベット・全年100%超）。確定払戻ベースの数字で、スリッページ込みでも小口なら生存。\n\n"
        f"⚠️ **実弾（前向き記録）は蓄積中**（毎朝自動・現在 {n} レース）。"
        f"{'サンプルが少なく統計的にはまだ無意味。' if n < 100 else ''}"
        "机上とライブの比較が言えるのは年単位（実際の回収率は「成績」ページ）。"
        "統計的推定であり的中・利益を保証しない。投資は自己責任で。",
        icon="📋",
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
    """持続センサー: 机上エッジが未来も通用してるかを前向き記録で監視（本日/成績 共通）。"""
    tracks = sensor.status().get("tracks", {})
    if tracks:
        summ = " ／ ".join(f"{_VMARK.get(t['verdict'], '⚪')}{_SHORT.get(tr, tr)}"
                           for tr, t in tracks.items())
        st.caption(f"🛰 **持続センサー**（机上エッジが未来も通用してるか）：{summ}"
                   "　🟢黒字 🟡失速入口 🔴崩れ ⚪判定保留(N不足)")
    with st.expander("🛰 持続センサー 詳細", expanded=expanded):
        st.caption("低分散の 単勝/2連単that持続カナリア。前向き記録で confidently 100%割れ(🔴)＝歪み消滅"
                   "＝3連複も道連れ。3連複はfat-tailで参考。判定は最小N到達後だけ（早合点を機械的に抑止）。")
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


def page_today():
    st.header("⚑ 本日の妙味レース")
    today = dt.date.today().strftime("%Y%m%d")
    cache = scan.load_cache()
    scanned_today = bool(cache and cache.get("date") == today)
    md = f"{today[4:6]}/{today[6:]}"

    # 本日のステータス（妙味ナシなら「無」を明示）
    if scanned_today and len(cache["picks"]) == 0:
        st.success(f"📅 **{md} 本日は妙味レース無し（見送り）**　"
                   "全レースでモデル本命がイン(1号艇)＝張らない日。")
    elif scanned_today:
        st.info(f"📅 **{md} 本日の妙味レース {len(cache['picks'])}件**")
    else:
        st.warning(f"本日({md})はまだスキャンされていません（毎朝9時に自動）。")

    render_sensor(expanded=False)   # 賭ける前に"未来も通用してるか"を一目で

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
    # 締切間際に押す: 2連単の締切オッズでEV判定（検証: EV>2.0だけ堅く300-500%）
    if unsettled and b2.button("🎯 今の2連単オッズでEV判定（EV>2.0が買い）"):
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
    c[1].metric("妙味レース", f"{len(eff)} 件")
    c[2].metric("結果確定", f"{ps['races']} 件")
    c[3].metric("合計収支(単勝+2連単)", f"{ps['total']['pl']:+,} 円")
    st.caption("各レース：単勝=本命／2連単=上位3点・各100円。結果はレース後およそ5分で自動反映（3分毎更新）。"
               "購入は自分でテレボート入力・自己責任。")

    live_odds = st.session_state.get("live_odds", {})
    st.divider()
    for r in eff:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            badge = " 🔥" if r.get("high_conf") else ""
            c1.markdown(f"### {r['venue']} {r['rno']}R{badge}")
            if r.get("high_conf"):
                c1.caption("🔥 高確信妙味（本命度が高い群）")
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

    # 参考: 無フィルタ土台(track A=机上188.7の直接対照)
    with st.expander("参考：無フィルタ土台（全妙味レース＝机上OOFの直接対照）"):
        st.caption(f"全 {ps['races']} レース記録。単勝 {roi(ps['tansho']['ret'], ps['tansho']['stake'])}"
                   f" / 2連単 {roi(ps['exacta']['ret'], ps['exacta']['stake'])}"
                   f" / 合計 {roi(ps['total']['ret'], ps['total']['stake'])}。"
                   "台帳は何を張ったかに関わらず全レース記録＝机上116/188.7の対照。上の主表示は単勝<1.5を見送った実運用版。")

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
    st.caption("※単勝1点＋2連単3点/レース・各100円の実トラック（track A＝無フィルタ土台の前向き記録）。"
               "2連単188.7%はファットテール込みで分散が大きく、エッジは高配当帯に薄く濃い。"
               "**有意化は年単位**（妙味は1日0〜2本・高配当帯的中は年に数十本）＝数十本で机上vsライブを判断せん。"
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
            f"⚑ **妙味レース・推奨買い目**　モデルがイン(1号艇)を否定 → "
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
**モデルの本命が1号艇「以外」になったレース＝『妙味レース』**だけを狙う。
そこは「公衆＝インに寄せる／モデル＝インを嫌う」がズレた場所で、非効率が残りやすい。

買い方は各レース **単勝（本命1点）＋2連単（確率上位3点）・各100円**。
さらに二段でフィルタ：**①単勝<1.5倍のレースは丸ごと見送り**（大本命＝荒れず単勝も2連単も控除率負け）、
**②2連単は締切間際のEV（＝確率×締切オッズ）が2.0超のときだけ買う**（後述）。
""")

    st.markdown("#### 検証で分かっていること（＝机上／過去データ）")
    c = st.columns(3)
    c[0].metric("単勝 回収率", "平均 116%", "3年OOF・各年108-121%")
    c[1].metric("2連単3点 回収率", "~188%", "OOF・CI[171,208]")
    c[2].metric("検証ベット数", "3,662", "全年100%超")
    st.markdown("""
戦略を選ぶのに一度も使っていない**3年(2023-25)で前進検証**（学習はテスト年より前だけ・補完リークなし・再最適化なし）
した結果が上の数字。**選択バイアス・小サンプル・偶然の一発**の疑いは、
ブートストラップ（3,000回リサンプルでCI[171,208]・100%割れ0%）と監査（結果参照なし・払戻結合の重複0）で潰した。
**この数字は「確定払戻」ベース**＝実際に確定した配当だけで計算しており、後述のオッズの穴の影響を受けない。
""")

    st.markdown("#### まだ分かっていないこと（正直に）")
    _live = "統計的に無意味" if n < 100 else "蓄積途上"
    st.markdown(f"""
- **ライブで同じ回収率が出るかは未証明**。上は全部過去データ。前向きの実記録（毎朝自動・現在 {n} レース）は
  まだ{_live}で、机上と比べて何か言えるのは**年単位**（下記）。
- **「なぜ高配当に寄ると儲かるか（割安拾い）」の“水準”は保留**。追検算で、過去オッズの一部（約29%）が
  確定配当とズレる（締切間際の駆け込み前の値）ことが分かり、その分の見かけの上乗せを差し引くと、
  **『モデルが市場より当てている』方向は全帯で残る（＝本物）が、その大きさは現データでは確定できない**。
  ※これは上の回収率（確定払戻ベース）とは別の話で、回収率自体は無傷。
- **戦略の作り方が特定2年に偏っていないか**は、別の期間データが増えないと詰めきれない。
""")

    st.markdown("#### 年間どれくらいのレースが対象か／『年単位』の意味")
    st.table(pd.DataFrame({
        "年": ["2023", "2024", "2025"],
        "総レース": ["55,272", "55,240", "55,066"],
        "妙味レース(本命≠1号艇)": ["1,254", "1,225", "1,183"],
        "出現率": ["2.3%", "2.2%", "2.1%"],
    }))
    st.markdown("""
**妙味レースは年 約1,200本**（全レースの約2.2%・1日ならし約3本、0本の日も普通）。単勝<1.5倍を丸ごと見送ると年 約1,000本。

本数は少なくないが、**賭式で分散がまるで違う**のが肝：
- **単勝は低分散** → 年1,200本あれば **約1年** で回収率が読めてくる
- **2連単は稀な高配当が牽引するファットテール** → 高配当帯は**年たった約84レース**しかなく、
  ブートで安定を見た3,662本（≒3年分）に並ぶまでCIが広いまま。**机上188%とライブを比べられるのは2〜3年**

だから「放っておいて積む」のが正解。中間集計の一喜一憂（今日勝った/負けた）は最大の敵で、見るなら四半期に一度。
""")

    st.markdown("#### 記録は3本立て（混同しないため）")
    st.markdown("""
- **track A＝無フィルタ土台**：妙味レースを全部記録した台帳。**机上188.7の直接の対照**（「成績」ページ）
- **reco＝フィルタ版（参考）**：単勝<1.5を見送った版。track Aから導く目安
- **実弾＝実際に張った分**：裁量・金額込みの別track

**どれも「土台188.7の検証」と同じではない**（母数も対象も違う）。特に実弾の成績を土台の証明と読まない。
""")

    st.markdown("#### 最新の状況（2026-07 更新）")
    st.markdown("""
- **2連単EVフィルタを実装**：締切間際のオッズで EV=確率×オッズ を出し、**EV>2.0のときだけ買う**。
  過去データでこの帯は堅い（点推定512% / 上位40本除外でも305% / ブートCI[435,597]・100%割れ0%）。
  ただし**巨大すぎ＝要ライブ確認**（成熟市場で巨大エッジは疑う）＋fat-tailでブートCIは見た目ほど締まってない。
  「本日の妙味レース」で🎯EVボタン→EV>2.0が🟢買い。
- **成績ページを「単勝<1.5見送り版」に**：実際に張る想定（チャラレース抜き）の回収率を主表示。無フィルタ土台は参考に格納。
- **特徴量を掃除（24→14）**：ROI-permutationで「効いてない軸」を三重確認して除去（回収率は不変）。
  分かったこと＝**儲けの本体は特徴でなく『妙味レースを選ぶこと（selection）』**。個々の選手の癖（生ID）を足しても伸びなかった。
- **自動化をクラウド（GitHub Actions）へ移設＝PC不要**：毎朝9時に自動でscan→結果反映→サイト更新。PCを切っても回る。
- **ライブ生死ラインを事前固定（プレレジ）**：数字を見てから基準を決めると人は都合よく解釈するので、
  「何本たまって回収△%を割ったら畳む／超えたら信じる」を先に固定済み（docs/LIVE_PREREG.md）。
  特にEV>2.0は**約300本(1年+)たまるまで良し悪しを判断しない**。
""")

    st.markdown("#### 使い方・免責")
    st.info(
        "半自動です（自動購入はしません）。「本日の妙味レース」の買い目をテレボート等へ**自分で入力**して投票。"
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
               f"※机上の月平均は+25万前後だが**ライブ未証明**。負け月も隠さず出す。")
    sc = AGENT_UNIT / 100
    ledger = papertrade._load()
    today = dt.date.today().strftime("%Y%m%d")

    # --- 今日の予想 ---
    st.markdown("#### 📅 今日の予想")
    todays = [r for r in ledger if r.get("date") == today and not r.get("settled")]
    if not todays:
        c = scan.load_cache()
        if c and c.get("date") == today and len(c.get("picks", [])) == 0:
            st.success("本日は妙味レース無し＝**全レース見送り**（プロは買わない日）。")
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

    st.info("※これは前向きの実記録＝ライブ検証。机上（過去+25万/月）が本当に出るかを積んで確かめる装置。"
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
        with st.expander(f"⚪ 見送り {len(mikoku)}レース（インが濃厚＝妙味なしと判断した回）"):
            for r in mikoku:
                st.caption(f"{r.get('deadline','')} {r['venue']}{r['rno']}R ｜ "
                           f"本命{r['honmei']}号({r.get('win_pct',0):.0f}%) 見送り")


def page_agent():
    st.header("🎙️ エージェント実況（全レース予想→勝負/見送り→結果）")
    st.caption("エージェントが本日の全レースを直前まで読み、妙味だけ勝負。結果that出たら一言。自動更新(3分毎)。"
               "※コメントはルール生成(LLM不使用)・儲け保証なし。実弾は自己責任。")
    log = commentary.load_log()
    today = dt.date.today().strftime("%Y%m%d")
    if not log or not log.get("races"):
        st.info("まだ本日のエージェントログがありません（毎朝の自動スキャンで生成されます）。")
        return
    if log.get("date") != today:
        st.caption(f"（直近スキャン: {log['date'][4:6]}/{log['date'][6:]} 分）")
    _agent_feed(log)


st.title("🚤 競艇予想")
status_banner()
page = st.sidebar.radio("ページ",
                        ["本日の妙味レース", "🎙️エージェント実況", "🌊荒れそうレース",
                         "成績（予想vs実際）", "プロの見方", "レース個別予想", "解説"])
st.sidebar.markdown("---")
st.sidebar.caption("愛知近郊: " + " / ".join(venue_name(j) for j in LOCAL_VENUES))
if page == "本日の妙味レース":
    page_today()
elif page == "🎙️エージェント実況":
    page_agent()
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
