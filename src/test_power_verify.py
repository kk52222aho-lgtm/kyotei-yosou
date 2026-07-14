"""#2=地力(実力)that本物か、それとも"強い選手は2号に入ってるだけ"の言い換えかを潰す。

test_when_inside_failsで地力38.6%>差し2号30.7だったが、今日の教訓=良い数字ほど疑う。
最大の疑い: 地力=ただの2号(差しの好位置)の言い換えではないか。3つの敵対検証:
  1. 位置を固定しても地力that効くか(各非イン枠の中で地力上位ほど勝つか=skill独立性)=本丸
  2. 地力ランク別に単調か(非イン地力1位→5位で勝率that綺麗に下がるか)
  3. 年別で#2安定か
前提はクリーン(6艇そろい・イン存在・イン負け)。survivorship除去済。確定払戻・walk-forward。

例: python -u -m src.test_power_verify
"""
from __future__ import annotations

import pandas as pd

from . import storage
from .validate import fit_predict_leakfree, SELECTION_YEARS


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.*, p.tansho_lane
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.tansho_lane IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def collect(df):
    rows = []
    for y in [y for y in sorted(df["yr"].unique()) if y not in SELECTION_YEARS]:
        te = (df["yr"] == y).to_numpy()
        tr = (df["yr"].astype(int) < int(y)).to_numpy()
        if tr.sum() < 3000 or te.sum() < 2000:
            continue
        test = fit_predict_leakfree(df, tr, te)
        for (date, jcd, rno), g in test.groupby(["date", "jcd", "rno"]):
            if len(g) != 6 or not (g["lane"] == 1).any():
                continue
            win_lane = int(g.iloc[0]["tansho_lane"])
            if win_lane == 1:                      # イン勝ちは除外(=位置が崩れた世界だけ)
                continue
            non1 = g[g["lane"] != 1].copy()
            non1["nrk"] = non1["nat_win"].rank(ascending=False, method="min")  # 非イン内の地力順位
            for r in non1.itertuples():
                rows.append({"yr": y, "lane": int(r.lane), "nrk": int(r.nrk),
                             "win": int(int(r.lane) == win_lane)})
    return pd.DataFrame(rows)


def wr(sub):
    return sub["win"].mean() * 100 if len(sub) else 0


def main():
    df = load()
    d = collect(df)
    print(f"インthaが負けたレースの非イン艇 総数 {len(d):,}（1レース5艇）\n")

    print("[検証2: 非イン内の地力ランク別 勝率(単調に下がれば実力that効いてる)]")
    for k in [1, 2, 3, 4, 5]:
        sub = d[d["nrk"] == k]
        print(f"  地力{k}位: 勝率 {wr(sub):4.1f}%  (n{len(sub)})")

    print("\n[検証1(本丸): 枠を固定して地力that効くか＝2号の言い換えでない証明]")
    print("  各枠の中で 地力上位(非イン1-2位) vs 下位(4-5位) の勝率差that出れば実力は位置と独立")
    for ln in [2, 3, 4, 5]:
        hi = d[(d["lane"] == ln) & (d["nrk"] <= 2)]
        lo = d[(d["lane"] == ln) & (d["nrk"] >= 4)]
        if len(hi) >= 40 and len(lo) >= 40:
            print(f"  {ln}号: 地力上位 {wr(hi):4.1f}% (n{len(hi)}) vs 地力下位 {wr(lo):4.1f}% (n{len(lo)})  "
                  f"差{wr(hi)-wr(lo):+4.1f}pt")
        else:
            print(f"  {ln}号: サンプル不足")

    print("\n[検証3: 年別 非イン地力1位の勝率(安定か)]")
    for y in sorted(d["yr"].unique()):
        sub = d[(d["yr"] == y) & (d["nrk"] == 1)]
        print(f"  {y}: 地力1位 勝率 {wr(sub):4.1f}%  (n{len(sub)})")

    print("\n判定: 検証2that単調・検証1で全枠プラス差・検証3で3年安定 なら #2=地力は本物(位置と独立の実力)。")
    print("      枠固定で差that消えるなら『地力=2号の言い換え』=偽物。確定払戻・walk-forward・6艇クリーン。")


if __name__ == "__main__":
    main()
