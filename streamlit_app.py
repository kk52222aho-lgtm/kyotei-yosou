"""競艇予想サイト（Streamlit 版）。

  streamlit run streamlit_app.py

既存の src モジュール（モデル・予測・スキャン）をそのまま利用する。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import predict, papertrade, scan, scraper
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
    return e


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

    date, rows = _latest_day_rows()
    if not rows:
        st.caption("まだ記録がありません。")
        return
    if date != today:
        st.divider()
        st.caption(f"↓ 直近の記録・結果（{date[4:6]}/{date[6:]}）")

    # 買う直前に押すと、現在の単勝オッズで1.5倍未満（明確な大本命）を弾く
    unsettled = [r for r in rows if not r.get("settled")]
    if unsettled and st.button("🔄 今の単勝オッズを取得して1.5倍判定"):
        live = {}
        with st.spinner("現在の単勝オッズを取得中…"):
            for r in unsettled:
                od = scraper.fetch_odds(r["date"], r["jcd"], r["rno"])
                live[f"{r['jcd']}-{r['rno']}"] = od.get(r["honmei"]) if od else None
        st.session_state["live_odds"] = live

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
            c1.markdown(f"### {r['venue']} {r['rno']}R")
            combos = " ・ ".join(r["exacta3"])
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
                    c2.markdown(f"**単勝 {r['honmei']}号**{odlbl} {nm} ／ "
                                f"2連単 上位3点 {combos}  \n⏳ 結果待ち")


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
    st.caption("↑ **track A＝無フィルタ土台のライブ検証**（全妙味レースを記録＝机上OOF 単勝116%/2連単188.7%の直接対照）。"
               "何を実際に張ったかに関わらず、台帳は全レースを記録する。")

    rc = ps.get("reco", {})
    if rc.get("stake"):
        st.info(f"💡 **参考：フィルタ版（単勝1.5倍未満を丸ごと見送り）**：単勝+2連単 合計 "
                f"回収率 **{rc['ret']/rc['stake']:.1%}**・収支 **{rc['pl']:+,}円**"
                f"（見送り {rc['skipped']}レース）。クリーン年検証：単勝<1.5は死に金(69%)。"
                f"※これはtrack Aから導いたフィルタ版の目安。実弾（裁量・金額込みで実際に張った分）はさらに別trackで、"
                f"**どちらも「土台188.7の検証」とは呼ばない**（母数も対象も違う）。")

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

    st.markdown("#### 使い方・免責")
    st.info(
        "半自動です（自動購入はしません）。「本日の妙味レース」の買い目をテレボート等へ**自分で入力**して投票。"
        "本サイトはシステム構築の研究目的で、実弾は少額（100円）想定。"
        "**統計的推定であり的中・利益を保証しません。投資は自己責任で。**"
    )


st.title("🚤 競艇予想")
status_banner()
page = st.sidebar.radio("ページ",
                        ["本日の妙味レース", "成績（予想vs実際）", "レース個別予想", "解説"])
st.sidebar.markdown("---")
st.sidebar.caption("愛知近郊: " + " / ".join(venue_name(j) for j in LOCAL_VENUES))
if page == "本日の妙味レース":
    page_today()
elif page == "成績（予想vs実際）":
    page_record()
elif page == "解説":
    page_about()
else:
    page_race()
