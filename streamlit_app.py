"""競艇予想サイト（Streamlit 版）。

  streamlit run streamlit_app.py

既存の src モジュール（モデル・予測・スキャン）をそのまま利用する。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import predict, papertrade, scan, slip
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

    with st.expander("📋 買い目スリップ（コピーして手入力）", expanded=True):
        unit = st.number_input("1点あたりの賭け額（円）", min_value=100, max_value=10000,
                               value=100, step=100)
        st.code(slip.format_slip({"date": date, "picks": rows}, int(unit)), language="text")
        st.caption("⚠️ 購入は自分でテレボートに入力（自動購入なし＝規約・金銭事故回避）。少額分散・自己責任。")

    st.divider()
    for r in rows:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {r['venue']} {r['rno']}R")
            body = (f"**単勝 {r['honmei']}号 {r.get('name') or ''}** ／ "
                    f"2連単 上位3点 {' ・ '.join(r['exacta3'])}")
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

    # 合計(単勝+2連単)の累積損益
    acc, cum = 0, []
    for r in settled:
        acc += ((r.get("tansho_return", 0) - 100)
                + (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100))
        cum.append(acc)
    st.line_chart(pd.DataFrame({"累積損益(合計・円)": cum}))

    tbl = []
    for r in reversed(settled):
        race_pl = ((r.get("tansho_return", 0) - 100)
                   + (r.get("exacta_return", 0) - r.get("exacta_points", 0) * 100))
        tbl.append({
            "日付": r["date"], "場": r["venue"], "R": r["rno"],
            "本命": f"{r['honmei']}号",
            "単勝": "🎯" if r.get("tansho_win") else "×",
            "2連単": "🎯" if r.get("exacta_win") else "×",
            "実結果(2連単)": r.get("exacta_result"),
            "レース収支": race_pl,
        })
    st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)
    st.caption("※単勝1点＋2連単3点/レース・各100円の実トラック（前向き記録）。過去の前進検証は単勝116%/2連単176%だが"
               "1日〜少数では大きくブレる（今日のように負ける日も普通）。数十〜百本の平均で判断。")


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
