"""競艇予想サイト（Streamlit 版）。

  streamlit run streamlit_app.py

既存の src モジュール（モデル・予測・スキャン）をそのまま利用する。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import predict, papertrade, scraper
from src.venues import VENUES, LOCAL_VENUES, name as venue_name

st.set_page_config(page_title="競艇予想", page_icon="🚤", layout="wide")


@st.cache_resource
def get_model():
    return predict.load_model()


LANE_COLOR = {1: "#ffffff", 2: "#000000", 3: "#e53935",
              4: "#1e88e5", 5: "#fdd835", 6: "#43a047"}


def status_banner():
    st.info(
        "✅ **前進検証クリア**: 戦略未使用の3年(2023-25)で単勝 平均116%（計3,662ベット・全年100%超）。"
        "スリッページ込みでも小口なら生存。\n\n"
        "⚠️ **実弾は前向き記録を蓄積中**（毎朝自動）。2連単+80〜95%の桁は実運用確認待ち。"
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


def page_today():
    st.header("⚑ 本日の妙味レース")
    date, rows = _latest_day_rows()
    if not rows:
        st.warning("本日の記録がまだありません（毎朝9時に自動記録されます）。")
        return

    ps = papertrade.portfolio_stats(rows)
    c = st.columns(4)
    c[0].metric("対象日", date)
    c[1].metric("妙味レース", f"{len(rows)} 件")
    c[2].metric("精算", f"{ps['races']} 件")
    c[3].metric("合計収支(単勝+2連単)", f"{ps['total']['pl']:+,} 円")

    st.caption("各レース：単勝=本命／2連単=上位3点、各100円・少額分散。購入は自分でテレボート入力・自己責任。")

    # 買う直前に押すと、現在の単勝オッズで1.5倍未満（明確な大本命）を弾く
    unsettled = [r for r in rows if not r.get("settled")]
    if unsettled and st.button("🔄 今の単勝オッズを取得して1.5倍判定"):
        live = {}
        with st.spinner("現在の単勝オッズを取得中…"):
            for r in unsettled:
                od = scraper.fetch_odds(r["date"], r["jcd"], r["rno"])
                live[f"{r['jcd']}-{r['rno']}"] = od.get(r["honmei"]) if od else None
        st.session_state["live_odds"] = live
    live_odds = st.session_state.get("live_odds", {})

    st.divider()
    for r in rows:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {r['venue']} {r['rno']}R")
            # 単勝オッズ＋低オッズフラグ（現在オッズ優先、無ければ朝オッズ）
            lo = live_odds.get(f"{r['jcd']}-{r['rno']}")
            o = lo if lo is not None else r.get("scan_odds")
            src_lbl = "現在" if lo is not None else "朝"
            if o and o < papertrade.ODDS_FLOOR:
                tansho = f"**単勝 {r['honmei']}号**（{src_lbl}{o:.1f}倍 ⚠低オッズ→見送り推奨）"
            elif o:
                tansho = f"**単勝 {r['honmei']}号**（{src_lbl}{o:.1f}倍）"
            else:
                tansho = f"**単勝 {r['honmei']}号**（オッズ未形成→上のボタンで取得）"
            body = f"{tansho} {r.get('name') or ''} ／ 2連単 上位3点 {' ・ '.join(r['exacta3'])}"
            if r.get("settled"):
                t = (f"🎯的中 {r.get('final_odds')}倍" if r.get("tansho_win")
                     else f"×ハズレ(勝ち{r.get('winner')}号)")
                e = (f"🎯的中 {r.get('exacta_return')}円" if r.get("exacta_win")
                     else f"×ハズレ(結果 {r.get('exacta_result') or '?'})")
                race_pl = ((r.get("tansho_return", 0) - 100)
                           + (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100))
                body += f"  \n単勝: {t}  ｜  2連単: {e}  ｜  **収支 {race_pl:+,}円**"
            else:
                body += "  \n⏳ 結果待ち"
            c2.markdown(body)


def page_record():
    st.header("📊 成績（予想 vs 実際）")
    ledger = papertrade._load()
    settled = sorted([r for r in ledger if r.get("settled")],
                     key=lambda r: (r["date"], r["jcd"], r["rno"]))
    if not settled:
        st.info("精算済みの記録がまだありません（毎朝9時に自動精算されます）。")
        return

    ps = papertrade.portfolio_stats(settled)
    c = st.columns(4)
    c[0].metric("確定レース", ps["races"])
    c[1].metric("単勝 回収率", f"{ps['tansho']['ret']/ps['tansho']['stake']:.1%}",
                f"{ps['tansho']['pl']:+,}円")
    c[2].metric("2連単 回収率", f"{ps['exacta']['ret']/ps['exacta']['stake']:.1%}" if ps['exacta']['stake'] else "―",
                f"{ps['exacta']['pl']:+,}円")
    c[3].metric("合計 回収率", f"{ps['total']['ret']/ps['total']['stake']:.1%}",
                f"{ps['total']['pl']:+,}円")

    hi = ps.get("tansho_hi", {})
    if hi.get("stake"):
        st.caption(f"💡 単勝オッズ1.5倍未満を見送った場合の単勝: "
                   f"{hi['races']}レース・回収率 {hi['ret']/hi['stake']:.1%}・収支 {hi['pl']:+,}円"
                   f"（低オッズは控除率負け＝クリーン年検証で回収率↑。締切オッズで判断）")

    # 合計(単勝+2連単)の累積損益
    acc, cum = 0, []
    for r in settled:
        acc += ((r.get("tansho_return", 0) - 100)
                + (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100))
        cum.append(acc)
    st.line_chart(pd.DataFrame({"累積損益(合計・円)": cum}))

    st.markdown("#### 明細（1点100円・単勝1点＋2連単3点）")
    tbl = []
    for r in reversed(settled):
        t_pl = r.get("tansho_return", 0) - 100
        e_stake = r.get("exacta_points", 0) * 100
        e_pl = r.get("exacta_return", 0) - e_stake
        od = f"{r.get('final_odds'):.1f}倍" if r.get("final_odds") else "―"
        tbl.append({
            "日付": r["date"],
            "レース": f"{r['venue']}{r['rno']}R",
            "単勝": f"{r['honmei']}号({od})",
            "単勝結果": (f"🎯 +{r.get('tansho_return', 0) - 100}円"
                        if r.get("tansho_win") else "× -100円"),
            "2連単 買い目": " ".join(r.get("exacta3", [])),
            "2連単 結果": (f"🎯 {r.get('exacta_result')} 払戻{r.get('exacta_return')}円"
                          if r.get("exacta_win")
                          else f"× 結果{r.get('exacta_result') or '?'} -{e_stake}円"),
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
    st.caption("※単勝1点＋2連単3点/レース・各100円の実トラック（前向き記録）。過去の前進検証は単勝116%/2連単176%だが"
               "1日〜少数では大きくブレる（負ける日も普通）。数十〜百本の平均で判断。")


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


st.title("🚤 競艇予想")
status_banner()
page = st.sidebar.radio("ページ", ["本日の妙味レース", "成績（予想vs実際）", "レース個別予想"])
st.sidebar.markdown("---")
st.sidebar.caption("愛知近郊: " + " / ".join(venue_name(j) for j in LOCAL_VENUES))
if page == "本日の妙味レース":
    page_today()
elif page == "成績（予想vs実際）":
    page_record()
else:
    page_race()
