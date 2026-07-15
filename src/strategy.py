"""probe戦略(旧「勝ちにいく統合戦略」): 唯一の+EV候補に極小で賭け、前向きに証明する。

【重要・5視点監査の結論(2026-07-15)】逆イン本命の歴史ROI110.9%は"勝ちモデル"ではない。
隔離した検死官/逆張り/張る側が独立に収斂: ①SE±6-10pp&的中CI±7-8%=110.9%はbreakevenと
統計的に区別不可(真値は約-3〜+11%) ②0.55閾値は40-60セル多重探索の生存者(p-hack)で
"OOS"は年を伏せただけで閾値を凍結してない ③イン/外はモデル確率で揃え市場オッズで揃えてない
=FLB(重本命の過小人気)の相乗りかもしれず"外"固有エッジは未分離 ④501本は選手/節でクラスタ=
実効N<<501。説明強要のみ機構(客はコースを値付け・薄い単勝プールで歪みが生存)を支持。
∴これは"未証明の仮説+妥当な機構"であって勝ちモデルではない。probeとして極小賭けで前向き収集し
実オッズで検証するまで信じない。増資禁止(自己インパクトでedge消滅)。

【証拠の等級(処刑側の統計も作文・即採点できる硬さで格付け)】
 硬い(棄却を単独で成立させる): ①バグ2個が実在=features._add_tenkaiのtest年リーク/
  validate.eval_yearの6艇フィルタ欠落(修正済・再現可) ②CI[94,129]が監査前から既にbreakevenを
  含んでいた(監査以前に確定した事実)。
 柔い(方向は妥当だが数字を確定扱いしない=粗探し縛りの視点that出した統計の語り):
  「40-60セルの多重比較」のセル数勘定・「p0.06-0.15」のp値。→「勝ちモデルでない」は硬い分だけで
  成立するので裁定は不変。だが柔い数字を鵜呑みにすると本物のエッジを潔癖で殺す工場になる=割引く。

【事前登録(FROZEN・以後この定義を再最適化しない)】
 ルール: モデル本命that lane!=1 かつ p0>=0.55 の単勝1点(P0_CORE=0.55で凍結)。
 母数: 前向き・実オッズ確定settleのfiresのみ(過去バックテストは母数に入れない=採点は未来だけ)。
 判定表(=先に書ける唯一の"未来"。数字でなく判決規則。racer_idクラスタbootstrapの生涯ROI-CIで):
   ・合格(卒業): N>=200 かつ CI下限>=100 → probe解除(なおも小さく)。機構が本物だった証。
   ・棄却(kill): (a)N>=100 で CI上限<100(自信を持ってbreakeven未満) or (b)N>=200 で 生涯ROI<90%
       → 帳簿の"死んだ側"に載せ、賭け停止。※延命禁止=合格ルールだけ持つと永遠に生かす工場になる。
   ・継続: 上記以外(CIが100を跨ぐ) → 最小賭けで採り続ける。
 再最適化の封印: P0_CORE等を動かしたくなったら、それは"新しい事前登録"で N をゼロから振り直す
  (貯めたfiresは再利用不可)。機構分割(star/地元/展示 vs 無名×モーター)は"記録だけ"=ここで閾値を選ばない。
 → 触らず走らせる。工場長は数字を先に書かない。書くのは檻(この判定表)だけ。

以下は元の設計メモ(数字は上記監査で割引くこと):
- 控除を超える+EVは「逆イン本命」だけ=本命≠1号(モデル1位が外)×確信p0≥.55 の単勝(≈109%,+9%)。
  歴史では点推定超だがCI[94,129]=未確定。真値100-109%の薄い本物。
- 他は全部控除負け: イン本命94%/大穴27%(FLBのカモ)/妙味exotic82%。→ 見送りが正解。
- 大穴(model_p低)は絶対張らない(FLB=27%)。展開込みモデルで本命的中57%(予測は最良)。

∴戦略= "ほとんど見送り、逆イン本命だけ張る"。賭け金は分数ケリー(エッジ薄+CI広→1/4ケリー)。
これは"確実に勝つ"でなく"唯一+EVの候補に、破産しない大きさで賭ける"実験的モデル。

使い方: decide(predict_entries(entries)) → 賭け判断。 python -m src.strategy でバックテスト。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- 逆イン本命エッジのパラメータ(test_flb_outside/oos 実測) ---
P0_CORE = 0.55        # 本命(確信)閾: これ以上=本命候補(単勝≈109%)
P0_SUB = 0.50         # 準候補(≈99.6%,breakeven付近)=極小 or 様子見
WINRATE = 0.645       # 逆イン本命の実勝率(p0≥.55)
AVG_ODDS = 1.69       # 平均勝ち配当(109%/64.5%)
KELLY_FRAC = 0.25     # 分数ケリー(未確定エッジ→控えめ)
UNIT_CAP = 0.04       # 1レース上限=資金の4%(破産回避)


def _kelly():
    b = AVG_ODDS - 1.0
    f = (b * WINRATE - (1 - WINRATE)) / b            # フルケリー
    return max(0.0, min(f * KELLY_FRAC, UNIT_CAP))   # 1/4ケリー・上限4%


def decide(rows: list[dict]) -> dict:
    """展開込みモデルのpredict_entries出力(rank順)から、勝ちにいく1つの賭け判断を返す。"""
    if not rows:
        return {"action": "SKIP", "reason": "出走表なし"}
    honmei = rows[0]
    lane = int(honmei["lane"])
    p0 = (honmei.get("win_pct") or 0) / 100.0
    stake = round(_kelly() * 100, 1)                  # 資金比%(表示用)

    if lane != 1 and p0 >= P0_CORE:
        return {
            "action": "BET", "bet": "単勝", "target": lane,
            "name": honmei.get("name"), "p0": round(p0, 3),
            "stake_pct": stake, "tier": "◎逆イン本命(probe)",
            "edge": "点推定+11%だが統計的にbreakevenと区別不可(5視点監査)",
            "reason": f"強い外本命({lane}号・確信{p0*100:.0f}%)。機構(客はコースを値付けし艇を"
                      f"過小評価)は妥当だが、歴史110.9%は多重探索+fat-tail+FLB相乗りでノイズと"
                      f"見分けられない=未証明の仮説。probeとして極小({stake:.1f}%)のみ・増資禁止。",
        }
    if lane != 1 and p0 >= P0_SUB:
        return {
            "action": "WATCH", "bet": "単勝", "target": lane,
            "name": honmei.get("name"), "p0": round(p0, 3),
            "stake_pct": round(stake / 2, 1), "tier": "○準・逆イン本命",
            "edge": "±0(breakeven付近)",
            "reason": f"外本命だが確信{p0*100:.0f}%＝歴史breakeven付近(99.6%)。張るなら極小、基本様子見。",
        }
    return {
        "action": "SKIP", "tier": "見送り",
        "reason": ("本命=イン(1号)＝市場が正しく値付け・控除負け(94%)。" if lane == 1
                   else "外本命だが確信不足＝エッジ帯外。") + "大穴は論外(FLB27%)。張らない。",
    }


# ---------------- バックテスト ----------------
def backtest():
    from . import storage
    from .validate import fit_predict_leakfree, SELECTION_YEARS

    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen FROM entries e JOIN payouts p
        ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]

    bets = []          # (yr, stake_units, ret_units)  1unit=ケリー基準の1レース
    races = 0
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6:
                continue
            races += 1
            g = g.sort_values("p", ascending=False)
            top = g.iloc[0]
            s = g["p"].sum()
            if s <= 0:
                continue
            rows = [{"lane": int(top["lane"]), "win_pct": float(top["p"]) / s * 100,
                     "name": None}]
            d = decide(rows)
            if d["action"] != "BET":
                continue
            if pd.isna(top["tansho_lane"]):
                continue
            won = int(top["tansho_lane"]) == d["target"]
            ret = float(top["tansho_yen"]) / 100.0 if won else 0.0    # 1円賭けの戻り
            bets.append((y, 1.0, ret, won))

    if not bets:
        print("賭け0"); return
    b = pd.DataFrame(bets, columns=["yr", "stake", "ret", "won"])
    n = len(b)
    roi = b["ret"].sum() / b["stake"].sum() * 100
    hit = b["won"].mean() * 100
    yrs = sorted(b["yr"].unique())
    per_yr = int(n / len(yrs))
    print("=== 勝ちにいく統合戦略 バックテスト(展開込みモデル・確定払戻・walk-forward・6艇クリーン) ===")
    print(f"  総レース {races:,} 中 賭けたの {n}件 ({n/races*100:.1f}%＝ほぼ見送り) / 年約{per_yr}本")
    print(f"  逆イン本命 単勝ROI {roi:.1f}%  的中{hit:.1f}%")
    for y in yrs:
        sy = b[b["yr"] == y]
        print(f"    {y}: {len(sy):4d}本 ROI{sy['ret'].sum()/len(sy)*100:5.1f}% 的中{sy['won'].mean()*100:.0f}%")
    stake_pct = _kelly() * 100
    exp = (roi - 100) / 100 * stake_pct
    print(f"\n  賭け金 1/4ケリー={stake_pct:.1f}%/レース(上限4%)。年約{per_yr}本×期待+{roi-100:.0f}%")
    print(f"  → 資金比の期待成長 ≈ 年 +{exp*per_yr:.1f}%相当(点推定ベース・CI[94,129]で0〜+も)")
    print("  ※控除を超える唯一の候補にだけ張り、他は見送り。未確定エッジゆえ小さく・破産しない大きさ。")


if __name__ == "__main__":
    backtest()
