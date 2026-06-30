"""競艇予想サイト（Streamlit 版）。

  streamlit run streamlit_app.py

既存の src モジュール（モデル・予測・スキャン）をそのまま利用する。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import predict, scan, slip
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


def page_today():
    st.header("⚑ 本日の妙味レース")
    cache = scan.load_cache()
    if not cache:
        st.warning("まだスキャンしていません。`python -m src.scan` を実行してください。")
        return
    st.caption(f"対象日 {cache['date']} / スキャン {cache['generated_at']} / "
               f"該当 {len(cache['picks'])} 件（モデルがイン=1号艇を否定したレース）")
    if not cache["picks"]:
        st.success("本日の妙味レースはありません（全レースでモデル本命がイン＝見送り推奨）。")
        return

    # 📋 買い目スリップ（テレボート手入力用・コピーボタン内蔵）
    with st.expander("📋 本日の買い目スリップ（コピーして手入力）", expanded=True):
        unit = st.number_input("1点あたりの賭け額（円）", min_value=100, max_value=10000,
                               value=100, step=100,
                               help="少額・分散が前提。プールを潰さない範囲で。")
        st.code(slip.format_slip(cache, int(unit)), language="text")
        st.caption("⚠️ 購入は自分でテレボートに入力（自動購入はしない＝規約・金銭事故リスク回避）。"
                   "1日の勝敗で判断せず、数十〜百本の平均で。投資は自己責任。")

    st.divider()
    for p in cache["picks"]:
        odds = f"{p['odds']:.1f}倍" if p.get("odds") else "—"
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {p['venue']} {p['rno']}R")
            c2.markdown(
                f"**単勝 {p['honmei']}号 {p['name']}**（予想{p['win_pct']}% / {odds}）  \n"
                f"2連単 上位3点: **{' ・ '.join(p['exacta3'])}**")


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
page = st.sidebar.radio("ページ", ["本日の妙味レース", "レース個別予想"])
st.sidebar.markdown("---")
st.sidebar.caption("愛知近郊: " + " / ".join(venue_name(j) for j in LOCAL_VENUES))
if page == "本日の妙味レース":
    page_today()
else:
    page_race()
