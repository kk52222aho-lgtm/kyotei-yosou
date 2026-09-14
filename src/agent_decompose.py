"""agent.py バックテストの3分解検証（レグ分解・オッズ乖離の物証・感度チェック）。

条件は agent.py と共通（UNIT円/点・SLIP控除・EV_TH）。既存 agent.collect() を流用。
結論を盛らない。判定不能なら判定不能と書く。

例: python -u -m src.agent_decompose
"""
from __future__ import annotations

import numpy as np

from .agent import collect, UNIT, SLIP, EV_TH
from .validate import load


def tan_pl(r):   # 単勝1レースP&L（円・slip込み）
    return (UNIT * r["tan_ret_per100"] * SLIP if r["won_t"] else 0) - UNIT


def ex_pl(r):    # 2連単3点1レースP&L（円・slip込み）
    return (UNIT * r["ex_ret_per100"] * SLIP if r["ex_win"] else 0) - UNIT * 3


def monthly(rows, plf):
    mo = {}
    for r in rows:
        mo.setdefault(r["date"][:6], 0.0)
        mo[r["date"][:6]] += plf(r)
    return mo


def main():
    df = load()
    rows = collect(df)
    bet = [r for r in rows if r["tan_odds"] >= 1.5]                 # 単勝レグ=agentのbet母集団
    evu = [r for r in bet if r["ev"] is not None and r["ev"] > EV_TH]  # 2連単レグ(EV>2.0)
    print(f"妙味 {len(rows)} / 単勝レグ(tan>=1.5) {len(bet)} / 2連単レグ(EV>2.0) {len(evu)}"
          f"（UNIT={UNIT}円・SLIP-{int((1-SLIP)*100)}%・EV_TH={EV_TH}）")

    # ===== 1. レグ分解 =====
    print("\n" + "=" * 62 + "\n【1. レグ分解】")
    n = len(bet)
    tan_raw = sum(r["tan_ret_per100"] for r in bet if r["won_t"]) / n
    tan_hit = sum(r["won_t"] for r in bet) / n
    tan_total = sum(tan_pl(r) for r in bet)
    tm = monthly(bet, tan_pl)
    print(f"■ 単勝レグ（tan>=1.5・{n}レース）")
    print(f"   raw ROI {tan_raw*100:.1f}% / slip後 {tan_raw*SLIP*100:.1f}% / 通算 {int(tan_total):+,}円 "
          f"/ 月平均 {int(np.mean(list(tm.values()))):+,}円 / 的中率 {tan_hit*100:.1f}%")
    # 健全性: フィルタ前(全妙味)の単勝raw ROI が検証済み115-118%と整合するか
    raw_all = sum(r["tan_ret_per100"] for r in rows if r["won_t"]) / len(rows)
    health = 112 <= raw_all * 100 <= 121
    print(f"   【健全性】全妙味の単勝raw ROI {raw_all*100:.1f}% → 検証済み115-118%と "
          f"{'整合✓' if health else 'ズレ⚠ agent.py要調査'}")

    m = len(evu)
    ex_raw = sum(r["ex_ret_per100"] for r in evu if r["ex_win"]) / (m * 3)
    ex_hit = sum(r["ex_win"] for r in evu) / m
    ex_total = sum(ex_pl(r) for r in evu)
    hits_pay = [r["ex_ret_per100"] * 100 for r in evu if r["ex_win"]]
    em = monthly(evu, ex_pl)
    print(f"■ 2連単レグ（EV>2.0・{m}レース）")
    print(f"   raw ROI {ex_raw*100:.0f}% / slip後 {ex_raw*SLIP*100:.0f}% / 通算 {int(ex_total):+,}円 "
          f"/ 月平均 {int(np.mean(list(em.values()))):+,}円")
    print(f"   的中率 {ex_hit*100:.1f}% / 的中時平均払戻 {int(np.mean(hits_pay)):,}円 / 的中数 {len(hits_pay)}")

    # ===== 2. オッズ乖離の物証 =====
    print("\n" + "=" * 62 + "\n【2. オッズ乖離の物証（勝ち組: キャッシュ vs 実払戻）】")
    hitr = [r for r in bet if r["ex_win"] and r.get("e_cache_odds") and r.get("e_actual")]

    def disc(r):
        return (r["e_cache_odds"] - r["e_actual"]) / r["e_actual"]

    sel = [r for r in hitr if r["ev"] > EV_TH]
    non = [r for r in hitr if r["ev"] <= EV_TH]
    print("的中レース限定・乖離率=(キャッシュオッズ-実オッズ)/実オッズ")
    stats = {}
    for name, grp in [("選別(EV>2.0)", sel), ("非選別(EV<=2.0)", non)]:
        d = np.array([disc(r) for r in grp]) if grp else np.array([0.0])
        stats[name] = d
        print(f"  {name}: n={len(grp):>4} 平均{d.mean()*100:>+7.1f}% 中央{np.median(d)*100:>+6.1f}% "
              f"25%tile{np.percentile(d,25)*100:>+6.1f}% 75%tile{np.percentile(d,75)*100:>+6.1f}% "
              f"最大{d.max()*100:>+7.0f}%")
    sel_mean = stats["選別(EV>2.0)"].mean() * 100
    sel_p75 = np.percentile(stats["選別(EV>2.0)"], 75) * 100

    # 決定打: 2連単レグの儲けは「クリーンオッズ(乖離<10%)」から来とるか「汚れ(≥10%)」からか
    print("\n--- 決定打: 2連単レグ収支をクリーン/汚れ勝ちレースで分解 ---")
    loss = sum(ex_pl(r) for r in evu if not r["ex_win"])   # 外れの-stakeは共通
    clean = [r for r in evu if r["ex_win"] and r.get("e_cache_odds") and r.get("e_actual")
             and abs(disc(r)) < 0.10]
    dirty = [r for r in evu if r["ex_win"] and r.get("e_cache_odds") and r.get("e_actual")
             and abs(disc(r)) >= 0.10]
    clean_ret = sum(UNIT * r["ex_ret_per100"] * SLIP - UNIT * 3 for r in clean)
    dirty_ret = sum(UNIT * r["ex_ret_per100"] * SLIP - UNIT * 3 for r in dirty)
    print(f"  外れ(全)の負担: {int(loss):+,}円")
    print(f"  クリーン的中(乖離<10%) {len(clean):>4}本 → 払戻ぶん {int(clean_ret):+,}円")
    print(f"  汚れ的中  (乖離>=10%) {len(dirty):>4}本 → 払戻ぶん {int(dirty_ret):+,}円")
    clean_leg = loss + clean_ret          # 汚れ的中を除いた=クリーンだけで賭けてた場合の近似
    print(f"  → 汚れ的中を除くと 2連単レグ通算 ≈ {int(clean_leg):+,}円（クリーンだけで儲かるか）")

    # ===== 3. 感度チェック（キャッシュオッズ割引→EV再計算→2連単レグ収支）=====
    print("\n" + "=" * 62 + "\n【3. 感度チェック（キャッシュオッズを割引→通算どこで消えるか）】")
    surv = {}
    for dsc in [0.0, 0.10, 0.20, 0.29]:
        th = EV_TH / (1 - dsc)              # 一律割引=EVがスケール→閾値が上がる
        u = [r for r in bet if r["ev"] is not None and r["ev"] > th]
        pl = sum(ex_pl(r) for r in u)
        surv[dsc] = pl
        print(f"  割引{int(dsc*100):>2}%（EV>{th:.2f}）: {len(u):>4}レース 通算 {int(pl):>+12,}円")

    # ===== 結論（盛らない）=====
    print("\n" + "=" * 62 + "\n【結論】")
    # 単勝
    if health:
        print(f"・単勝レグは信用できる：全妙味raw ROI {raw_all*100:.1f}%が検証済み115-118%と整合。leg収支{int(tan_total):+,}円。")
    else:
        print(f"・単勝レグは要調査：全妙味raw ROI {raw_all*100:.1f}%が115-118%からズレ＝agent.collectのバグ疑い。")
    # 2連単: 決定打は「汚れ的中を除いても儲かるか」＋選別群の乖離裾(中央でなく平均/75%tile)
    #  ※一律割引(Part3)は濃縮乖離を除去できず誤誘導するので主根拠にしない
    sel_inflated = sel_p75 > 50 or sel_mean > 50    # 選別群の上側裾が系統的に大きい
    if clean_leg <= 0:
        print(f"・2連単レグの利益はノイズ寄り（artifact）：汚れ的中(キャッシュ過大)を除くと通算≈{int(clean_leg):+,}円"
              f"＝儲けは締切スナップショットが勝ち組を過大記録した選別に依存。選別群乖離 平均{sel_mean:+.0f}%/75%tile{sel_p75:+.0f}%（非選別ほぼ0）。")
    elif not sel_inflated:
        print(f"・2連単レグはエッジ寄り：汚れ除いても通算{int(clean_leg):+,}円が残り、選別群の乖離裾も小さい。")
    else:
        print(f"・2連単レグは要警戒/判定不能：汚れ除くと{int(clean_leg):+,}円は残るが、選別群乖離 平均{sel_mean:+.0f}%/75%tile{sel_p75:+.0f}%"
              f"が非選別より系統的に大きい＝機上はスナップショットartifactで釣り上がっとる。ライブ(真の締切odズ)で要再判定。")
    print("\n※一律割引(Part3)は生存したが、乖離は一律でなく特定レースに濃縮＝Part3は誤った感度テスト。決定打は汚れ分解。")
    print("※全て机上（過去・締切オッズ29%乖離込み・ライブ未証明）。")


if __name__ == "__main__":
    main()
