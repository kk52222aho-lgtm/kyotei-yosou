"""経験学習エージェント v2（v1＋地合い＋崩れ条件＝統計が構造上持てん文脈）。

v1の結論: crude signature(場/艇番/級/勝率帯)の経験記憶はbase rate再生産で「1号艇に張れ」に
  収束(contrarian 2.1%)、学習曲線フラット。エッジ(本命≠1号艇)に触れてすらいない。

v2の狙い: 「いつ1号艇が飛ぶか」を経験で覚えさせ、エージェント自身にcontrarianを選ばせる。
  そのため記憶を【統計モデルが1レース独立では表現できない文脈】で条件付ける:
    - 地合い(ground): その水面(date,jcd)で"今日ここまで"の1号艇勝率→荒れ/普通/堅い。逐次状態。
    - 崩れ条件: 風速バケット / 1号艇がB級か。
  階層バックオフ: 詳細状況が密なら使い、疎なら粗い状況へ後退＝経験の汎化。

リーク規律(v1と同じ厳格さ):
  - 記憶更新は必ず予測の後。学習=2022-2024で育て、記憶を凍結して未見2025を1発勝負。
  - 地合いは当日その水面の"過去レースだけ"から算出＝観測可能な文脈(学習パラメータでない)。
    両フェーズでライブ計算してよい(未来を見てない)。

審判: contrarian率が2.1%から上がるか / 1着的中が1号艇ベタ(54.4%)+v1(+0.7pt)を超えるか /
  エージェントがcontrarianと言った時、実際に1号艇が飛んでるか(precision) / そこの3連単ROIが壁を破るか。

例: python -u -m src.experiential_v2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage
from .features import CLASS_MAP
from .strategy_search import trifecta_probs

ALPHA = 0.5
MIN_N = 40                       # バックオフ: このn未満のセルは使わず粗い状況へ後退
LEARN_YEARS = {"2022", "2023", "2024"}
TEST_YEAR = "2025"


def load():
    conn = storage.connect()
    df = pd.read_sql_query("""
        SELECT e.date, e.jcd, e.rno, e.lane, e.racer_class, e.nat_win,
               e.wind_speed, e.wave_height,
               p.trifecta_combo, p.trifecta_yen
        FROM entries e JOIN payouts p ON e.date=p.date AND e.jcd=p.jcd AND e.rno=p.rno
        WHERE e.win IS NOT NULL AND p.trifecta_yen IS NOT NULL
    """, conn)
    conn.close()
    df["yr"] = df["date"].str[:4]
    return df


def cls_num(c):
    return CLASS_MAP.get(str(c).strip(), 0)


def nat_b(v):
    return int(round(v)) if pd.notna(v) else -1


def wind_b(v):
    if pd.isna(v):
        return -1
    if v < 2:
        return 0        # 凪
    if v < 5:
        return 1        # 中風
    return 2            # 強風(荒れやすい)


def ground_b(g):
    """その水面の今日ここまでの1号艇勝率→地合いバケット。samples少なら不明(-1)。"""
    n, w = g
    if n < 4:
        return -1
    r = w / n
    if r < 0.40:
        return 0        # イン受難デー(荒れ)
    if r <= 0.60:
        return 1        # 普通
    return 2            # イン堅いデー


def levels(jcd, lane, cls, nb, wb, gb, l1weak):
    """具体→抽象の階層キー。文脈(風/地合い/1号艇弱)を長く残す並び。"""
    return [
        (jcd, lane, cls, nb, wb, gb, l1weak),
        (jcd, lane, cls, wb, gb, l1weak),
        (lane, cls, wb, gb, l1weak),
        (lane, wb, gb, l1weak),
        (lane, wb, gb),
        (lane, gb),
        (lane,),
    ]


def win_prop(mem, keys):
    """バックオフ: n>=MIN_N の最も具体的なセルの平滑化P(1着)。無ければ一様。"""
    for k in keys:
        r = mem.get(k)
        if r and r[0] >= MIN_N:
            return (r[1] + ALPHA) / (r[0] + 4 * ALPHA)
    return 0.25


def build_race(g):
    """1レース→ boats=[(lane, cls, nat_b)], 風b, 1号艇弱flag, combo, yen。"""
    boats = [(int(r.lane), cls_num(r.racer_class), nat_b(r.nat_win)) for r in g.itertuples()]
    wb = wind_b(g.iloc[0]["wind_speed"])
    l1 = [b for b in boats if b[0] == 1]
    l1weak = 1 if (l1 and l1[0][1] <= 2) else 0    # 1号艇がB級(<=B1)なら弱
    top = g.iloc[0]
    return {"jcd": str(g.iloc[0]["jcd"]), "boats": boats, "wb": wb, "l1weak": l1weak,
            "combo": top["trifecta_combo"], "yen": float(top["trifecta_yen"])}


def belief(mem, race, gb):
    """各艇の正規化P(1着)。文脈gb(地合い)・wb(風)・l1weak込みのバックオフ記憶から。"""
    jcd, wb, l1w = race["jcd"], race["wb"], race["l1weak"]
    props = {}
    for lane, cls, nb in race["boats"]:
        props[lane] = win_prop(mem, levels(jcd, lane, cls, nb, wb, gb, l1w))
    tot = sum(props.values())
    if tot <= 0:
        return {l: 1.0 / len(race["boats"]) for l, _, _ in race["boats"]}
    return {l: v / tot for l, v in props.items()}


def update(mem, race, gb, combo):
    jcd, wb, l1w = race["jcd"], race["wb"], race["l1weak"]
    parts = str(combo).split("-")
    pos = {}
    if len(parts) == 3:
        for i, l in enumerate(parts):
            try:
                pos[int(l)] = i + 1
            except ValueError:
                pass
    for lane, cls, nb in race["boats"]:
        p = pos.get(lane, 0)
        for k in levels(jcd, lane, cls, nb, wb, gb, l1w):
            r = mem.setdefault(k, [0, 0, 0, 0])
            r[0] += 1
            if p in (1, 2, 3):
                r[p] += 1


def run_phase(df, mem, ground, learn, use_ground=True):
    """1レースずつ: 地合い読取→予測→(学習期のみ)記憶更新→地合い更新。統計を返す。
    use_ground=False: 地合いbucketを定数化=情報を抜く(アブレーション)。他は完全に同一。"""
    df = df.sort_values(["date", "jcd", "rno"])
    recs = []
    online = []
    for (date, jcd, rno), g in df.groupby(["date", "jcd", "rno"], sort=False):
        race = build_race(g)
        gkey = (date, jcd)
        gstate = ground.setdefault(gkey, [0, 0])
        gb = ground_b(gstate) if use_ground else -99   # ★地合い(定数化で情報抜き)
        bel = belief(mem, race, gb)
        pick1 = max(bel, key=bel.get)
        actual1 = int(str(race["combo"]).split("-")[0])
        if learn:
            update(mem, race, gb, race["combo"])    # ★予測の後に学習
            online.append(pick1 == actual1)
        else:
            recs.append({"pick1": pick1, "actual1": actual1,
                         "belief": bel, "combo": str(race["combo"]), "yen": race["yen"],
                         "gb": gb})
        # 地合い更新(予測後・両フェーズ=観測可能文脈)
        gstate[0] += 1
        if actual1 == 1:
            gstate[1] += 1
    return recs, online


def experiment(df, use_ground):
    """with/without 地合い で新規学習→記憶凍結→未見テスト。指標dictを返す。"""
    mem, ground = {}, {}
    run_phase(df[df["yr"].isin(LEARN_YEARS)], mem, ground, learn=True, use_ground=use_ground)
    recs, _ = run_phase(df[df["yr"] == TEST_YEAR], mem, {}, learn=False, use_ground=use_ground)
    n = len(recs)
    lane1_base = np.mean([r["actual1"] == 1 for r in recs])
    agent_hit = np.mean([r["pick1"] == r["actual1"] for r in recs])
    contr = [r for r in recs if r["pick1"] != 1]

    # contrarian 3連単6点 ROI と fat-tail(上位30本除外)
    K = 6
    rets = []
    for r in contr:
        picks = [c for c, _ in trifecta_probs(r["belief"])][:K]
        rets.append(r["yen"] if r["combo"] in picks else 0.0)
    rets = np.sort(np.array(rets))[::-1] if rets else np.array([0.0])
    stake = max(len(contr), 1) * K * 100
    return {
        "cells": len(mem), "n": n,
        "lane1_base": lane1_base, "agent_hit": agent_hit,
        "contr_rate": len(contr) / n, "n_contr": len(contr),
        "break_prec": np.mean([r["actual1"] != 1 for r in contr]) if contr else float("nan"),
        "exact": np.mean([r["pick1"] == r["actual1"] for r in contr]) if contr else float("nan"),
        "roi6": rets.sum() / stake,
        "roi6_x30": rets[30:].sum() / stake,
        "hits": int((rets > 0).sum()),
    }


def main():
    df = load()
    print(f"3連単払戻あり {df.groupby(['date','jcd','rno']).ngroups:,}レース")
    print("アブレーション: v2から【地合い(日レベル逐次状態)】だけ抜いて比較（他は完全同一）")
    print("問い: 地合いは 風/級/場/艇番 の静的特徴(≒HGBが持つ情報)に上乗せしてるか？\n")

    on = experiment(df, use_ground=True)
    off = experiment(df, use_ground=False)

    base_break = 1 - on["lane1_base"]
    rows = [
        ("記憶セル数", f"{on['cells']:,}", f"{off['cells']:,}", ""),
        ("1着的中率", f"{on['agent_hit']*100:.1f}%", f"{off['agent_hit']*100:.1f}%",
         f"1号艇ベタ {on['lane1_base']*100:.1f}%"),
        ("contrarian率", f"{on['contr_rate']*100:.1f}%", f"{off['contr_rate']*100:.1f}%", "v1=2.1%"),
        ("1号艇飛び検知", f"{on['break_prec']*100:.1f}%", f"{off['break_prec']*100:.1f}%",
         f"ベース {base_break*100:.1f}%"),
        ("本命ピタリ1着", f"{on['exact']*100:.1f}%", f"{off['exact']*100:.1f}%", ""),
        ("contr 3連単6点ROI", f"{on['roi6']*100:.0f}%", f"{off['roi6']*100:.0f}%", ""),
        ("  上位30本除外後", f"{on['roi6_x30']*100:.0f}%", f"{off['roi6_x30']*100:.0f}%", "fat-tail除去床"),
        ("contr件数/的中", f"{on['n_contr']:,}/{on['hits']}", f"{off['n_contr']:,}/{off['hits']}", ""),
    ]
    print(f"  {'指標':<20}{'地合いあり':>12}{'地合いなし':>12}   参考")
    for name, a, b, note in rows:
        print(f"  {name:<20}{a:>12}{b:>12}   {note}")

    # 判定: 地合いを抜いてcontrarian検知/ROIがどれだけ落ちるか
    d_contr = (on["contr_rate"] - off["contr_rate"]) / on["contr_rate"] * 100 if on["contr_rate"] else 0
    d_prec = (on["break_prec"] - off["break_prec"]) * 100
    print("\n--- 判定 ---")
    print(f"  地合いを抜くと: contrarian率 {d_contr:+.0f}% / 1号艇飛び検知 {d_prec:+.1f}pt")
    if abs(d_prec) < 3 and abs(d_contr) < 15:
        print("  → ほぼ不変 = 地合いは上乗せしてない。静的特徴(風/級/場)で足りてた＝")
        print("     『statが構造上持てん逐次文脈』ですら壁を動かさん。方法非依存を完全決着。")
    else:
        print("  → 有意に低下 = 地合い(日レベル状態)が本体。statが持てん文脈が効いた証拠＝")
        print("     経験学習が静的特徴を超えた唯一の実例。3(検知器→単/2連)の芽が残る。")


if __name__ == "__main__":
    main()
