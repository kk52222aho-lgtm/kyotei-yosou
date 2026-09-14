"""経験学習エージェント v1（統計モデル無し・オンライン・episodic memory）。

思想: 14特徴の勾配フィットでなく、過去を1レースずつ実際に張り、結果で記憶を更新して
  経験で賢くなる。その日の状況・場の癖を"考えながら"学ぶ、の最小honest版。

仕組み(モデル一切不使用):
  - 各艇の【レース前だけで分かる状況】signature = (場, 艇番, 級, 全国勝率帯)。
  - 記憶 M[signature] = その状況の艇が過去に実際 1/2/3着 だった回数（オンライン累積）。
  - 予測: 今ある M だけから各艇の P(1着) を出し、正規化→Harvilleで3連単確率。
  - 1レースごと: 予測(=Mの現在値のみ) → 張る → 結果を見る → M を更新 → 次へ。
    ★更新は必ず予測の後 = 未来リーク無し・時系列厳守。

「外した記憶を持ってやり直す」の honest 解釈:
  同じレースを当たるまで繰り返す(=答えの丸暗記=最悪リーク)のでなく、
  その"状況クラス"の教訓(M更新)を次の似たレースに持ち越す。転移する唯一の形。

正直テスト:
  学習期(2022-2024)で M を育てる → M 凍結 → 未見の 2025 を1発勝負で走らせ ROI/的中を測る。
  壁は市場: 3連単は控除率+組合せ爆発。経験学習は"エッジの在処"を変えない。ここで確認する。

例: python -u -m src.experiential
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .features import CLASS_MAP
from .strategy_search import trifecta_probs

ALPHA = 0.5          # ディリクレ平滑化(未経験signatureはP一様=弱く張る)
LEARN_YEARS = {"2022", "2023", "2024"}
TEST_YEAR = "2025"


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.racer_class, e.nat_win,
               p.trifecta_combo, p.trifecta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def sig(jcd, lane, racer_class, nat_win):
    """レース前だけで決まる状況シグネチャ(結果を一切含まない=リーク不可)。"""
    cls = CLASS_MAP.get(str(racer_class).strip(), 0)
    nb = int(round(nat_win)) if pd.notna(nat_win) else -1   # 全国勝率帯 ~3..8
    return (str(jcd), int(lane), cls, nb)


def win_prop(mem, s):
    """記憶から P(この状況の艇が1着) を平滑化して返す。未経験なら一様寄り。"""
    r = mem.get(s)
    if r is None:
        return ALPHA / (0 + 4 * ALPHA)      # =1/4 一様
    n, c1, _, _ = r
    return (c1 + ALPHA) / (n + 4 * ALPHA)


def race_belief(mem, boats):
    """boats=[(lane, sig)] → 各艇の正規化P(1着) dict{lane:p}。経験ゼロなら全艇一様。"""
    props = {lane: win_prop(mem, s) for lane, s in boats}
    tot = sum(props.values())
    if tot <= 0:
        return {lane: 1.0 / len(boats) for lane, _ in boats}
    return {lane: v / tot for lane, v in props.items()}


def update(mem, boats, combo):
    """結果(trifecta_combo='a-b-c')でMを更新。各艇の実着(1/2/3/圏外)を累積。"""
    parts = str(combo).split("-")
    pos = {}
    if len(parts) == 3:
        for i, l in enumerate(parts):
            try:
                pos[int(l)] = i + 1
            except ValueError:
                pass
    for lane, s in boats:
        r = mem.setdefault(s, [0, 0, 0, 0])
        r[0] += 1                              # n
        p = pos.get(lane, 0)
        if p in (1, 2, 3):
            r[p] += 1                          # c1/c2/c3


def race_iter(df):
    """(date,jcd,rno)ごとに boats=[(lane,sig)] と実結果を時系列順で。"""
    df = df.sort_values(["date", "jcd", "rno"])
    for (date, jcd, rno), g in df.groupby(["date", "jcd", "rno"], sort=False):
        boats = [(int(r.lane), sig(r.jcd, r.lane, r.racer_class, r.nat_win))
                 for r in g.itertuples()]
        top = g.iloc[0]
        yield {
            "date": date, "boats": boats,
            "combo": top["trifecta_combo"], "yen": float(top["trifecta_yen"]),
        }


def main():
    df = load()
    print(f"3連単払戻あり {df.groupby(['date','jcd','rno']).ngroups:,} レース "
          f"(学習{sorted(LEARN_YEARS)} / テスト{TEST_YEAR})")

    mem = {}
    # ===== 学習期: 1レースずつ 予測→張る→結果→記憶更新（オンライン学習曲線も記録）=====
    learn = df[df["yr"].isin(LEARN_YEARS)]
    races = list(race_iter(learn))
    print(f"\n=== 学習期 {len(races):,}レースを時系列に経験 ===")
    hits, seen = [], 0
    for r in races:
        belief = race_belief(mem, r["boats"])           # ★更新前のMだけで予測
        pick1 = max(belief, key=belief.get)
        actual1 = int(str(r["combo"]).split("-")[0])
        hits.append(pick1 == actual1)
        update(mem, r["boats"], r["combo"])             # ★予測の後に学習
        seen += 1
    hits = np.array(hits)
    # 学習曲線: 10分割で1着的中率が上がるか(経験が効いてるか)
    print("  オンライン1着的中率(経験の蓄積で上がるか):")
    chunks = np.array_split(hits, 10)
    print("   " + " ".join(f"{c.mean()*100:4.1f}" for c in chunks) + "  (%/10分割・左=経験浅)")
    print(f"  記憶した状況パターン数: {len(mem):,}")

    # ===== テスト期: 記憶を凍結して未見の2025を1発勝負 =====
    test = df[df["yr"] == TEST_YEAR]
    tr = list(race_iter(test))
    print(f"\n=== テスト期 {TEST_YEAR}（記憶凍結・未見・1発勝負）{len(tr):,}レース ===")
    n = len(tr)
    lane1_base = np.mean([int(str(r["combo"]).split("-")[0]) == 1 for r in tr])
    agent_hit = 0
    # 3連単 top-N ROI（100円/点）。1着一致も別途。
    Ns = [1, 3, 6, 12, 24]
    tf_ret = {k: 0.0 for k in Ns}
    tf_hit = {k: 0 for k in Ns}
    contr = 0
    for r in tr:
        belief = race_belief(mem, r["boats"])
        pick1 = max(belief, key=belief.get)
        actual1 = int(str(r["combo"]).split("-")[0])
        agent_hit += (pick1 == actual1)
        contr += (pick1 != 1)
        ranked = [c for c, _ in trifecta_probs(belief)]
        for k in Ns:
            picks = ranked[:k]
            if str(r["combo"]) in picks:
                tf_hit[k] += 1
                tf_ret[k] += r["yen"]

    print(f"  1着的中: エージェント {agent_hit/n*100:.1f}%  vs  1号艇ベタ {lane1_base*100:.1f}%  "
          f"(差 {(agent_hit/n-lane1_base)*100:+.1f}pt)")
    print(f"  エージェントが本命≠1号艇と判断: {contr/n*100:.1f}%")
    print(f"\n  3連単 top-N（100円/点・確定払戻）:")
    print(f"  {'N点':>4}{'的中率':>9}{'回収率':>9}")
    for k in Ns:
        roi = tf_ret[k] / (k * 100 * n)
        print(f"  {k:>4}{tf_hit[k]/n*100:>8.1f}%{roi*100:>8.1f}%")
    print("\n※回収率<100%が期待値(市場の壁=控除25%+組合せ爆発)。経験学習でも在処は変わらん。")
    print("※これはv1(モデル無し・signature=場/艇番/級/勝率帯のみ)。地合い(日レベル状態)・展示・風は未搭載。")
    print("※次レバー: ①1着増分が出るか(出た所だけ深掘る) ②signatureに地合い/展示を足す(統計が持てん文脈)。")


if __name__ == "__main__":
    main()
