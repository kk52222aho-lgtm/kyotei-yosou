"""展開の+1.2pt(1着精度)は"金"か"市場が既に値付けした精度"か＝本物の軸で決着させる。

test_tenkaiで展開(6艇相対)that初めて1着精度を+1.2pt動かした。だが妙味単勝は-17pt=落ちた。
仮説: 展開(展示ST/タイム)は客も見る公開情報→モデルthaが群衆に近づき精度up・逆張りエッジdown
     =市場効率の壁と整合(上がった分は既に値付け済)。
決着=1着精度でなく本物の金軸で測る: 本命≠1号レースの2連単3点ROI(既存183-195%の本戦略)を
     ベース vs +展開 で年別に。展開that本当のエッジ源なら2連単3点も上がるはず。落ちれば
     "精度だけ(市場が値付け済)"で確定。確定払戻・walk-forward・リークなし。

例: python -u -m src.test_tenkai2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import storage
from .features import FEATURES, build_frame
from .train import _make_estimator
from .validate import SELECTION_YEARS
from .test_tenkai import add_tenkai, TENKAI


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane, p.tansho_yen, p.exacta_combo, p.exacta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_yen IS NOT NULL AND p.exacta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def eval_year(df, extra, yv):
    cols = FEATURES + extra
    base = build_frame(df, impute=False)[FEATURES].reset_index(drop=True)
    X = pd.concat([base, df[TENKAI].reset_index(drop=True)], axis=1)[cols].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=int)
    te = (df["yr"] == yv).to_numpy()
    tr = (df["yr"].astype(int) < int(yv)).to_numpy()
    if tr.sum() < 3000 or te.sum() < 2000:
        return None
    m = CalibratedClassifierCV(_make_estimator(), method="isotonic", cv=3)
    m.fit(X[tr], y[tr])
    out = df[te].copy()
    out["p"] = m.predict_proba(X[te])[:, 1]
    acc = []
    tan_n, tan_ret = 0, 0.0            # 妙味単勝
    ex_n, ex_ret = 0, 0.0             # 妙味2連単3点
    for _, gg in out.groupby(["date", "jcd", "rno"]):
        gg = gg.sort_values("p", ascending=False)
        top = gg.iloc[0]
        if pd.isna(top["tansho_lane"]):
            continue
        honmei = int(top["lane"])
        acc.append(int(top["tansho_lane"]) == honmei)
        if honmei == 1:
            continue
        tan_n += 1
        if int(top["tansho_lane"]) == honmei:
            tan_ret += float(top["tansho_yen"])
        second = [int(l) for l in gg["lane"] if int(l) != honmei][:3]
        buys = {f"{honmei}-{s}" for s in second}
        ex_n += 1
        if str(top["exacta_combo"]) in buys:
            ex_ret += float(top["exacta_yen"])
    return {
        "yr": yv, "acc": np.mean(acc) * 100,
        "tan": tan_ret / (tan_n * 100) * 100 if tan_n else 0, "tan_n": tan_n,
        "ex": ex_ret / (ex_n * 3 * 100) * 100 if ex_n else 0, "ex_n": ex_n,
    }


def main():
    df = load()
    df = add_tenkai(df)
    years = [v for v in sorted(df["yr"].unique()) if v not in SELECTION_YEARS]
    print("            |      1着精度       |  妙味単勝ROI   |   妙味2連単3点ROI(本戦略)")
    print("  年   件数 | ベース +展開   Δ   | ベース +展開   | ベース +展開     Δ")
    agg = {"ab": [], "ap": [], "eb": [], "ep": [], "en": 0}
    for yv in years:
        b = eval_year(df, [], yv)
        p = eval_year(df, TENKAI, yv)
        if not b or not p:
            continue
        print(f"  {yv} {b['ex_n']:5d}| {b['acc']:5.1f} {p['acc']:5.1f} {p['acc']-b['acc']:+5.1f} "
              f"| {b['tan']:5.0f} {p['tan']:5.0f}   | {b['ex']:5.0f} {p['ex']:5.0f} {p['ex']-b['ex']:+6.0f}")
        agg["ab"].append(b["acc"]); agg["ap"].append(p["acc"])
        # bet件数はbaseと+展開で違う→各々の自件数で割る(分母ミスマッチ禁止)
        agg["eb"].append((b["ex"] * b["ex_n"], b["ex_n"]))
        agg["ep"].append((p["ex"] * p["ex_n"], p["ex_n"]))
    print("  " + "-" * 66)
    dacc = np.mean(agg["ap"]) - np.mean(agg["ab"])
    eb = sum(r for r, _ in agg["eb"]) / sum(n for _, n in agg["eb"])
    ep = sum(r for r, _ in agg["ep"]) / sum(n for _, n in agg["ep"])
    print(f"  平均 1着Δ {dacc:+.1f}pt  |  妙味2連単3点(加重) ベース{eb:.0f}% -> +展開{ep:.0f}%  Δ{ep-eb:+.0f}pt")
    print()
    if ep - eb >= 8:
        print("  ** 本物の金軸(2連単3点)も展開で上がった＝展開はエッジ源。個別検証(EV/持続)へ。")
    elif dacc >= 0.5 and ep - eb < 5:
        print("  展開は1着精度を上げるthat本物のエッジ(2連単3点)は上げない/むしろ削る")
        print("  ＝仮説どおり『展開は公開情報で市場that既に値付け済、精度up分はエッジにならん』で確定。")
        print("  ＝壁はより賢い予測では破れん・エッジは群衆との食い違い(逆張り)にしか無い、を再確認。")
    else:
        print("  展開の1着精度Δも年で不安定＝ノイズ寄り。")
    print("\n※確定払戻・walk-forward・リークなし。")


if __name__ == "__main__":
    main()
