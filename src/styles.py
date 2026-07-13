"""買い方くらべ: 賭式×点数のスタイル別トータル。

HISTORICAL = test_styles.py の結果（妙味 本命≠1号・クリーン年3,612R・1点100円・確定払戻）。
  ＝「こう買ったら3年でトータルこう」の本体（単日は2レース程度でノイズなので履歴that主）。
compute() = 1レースの各スタイル収支（本日ぶんの実例表示用）。
※確定払戻ベース＝ライブ未証明。点を増やすほど的中率↑・万舟捕捉↑だが回収率↓（控除）。
"""
from __future__ import annotations

# (表示名, 単勝点, 2連単点, 3連複点, 3連単点)
STYLES = [
    ("堅実 4点（単1+2連3）", 1, 3, 0, 0),
    ("標準 11点（+3複4+3単3）", 1, 3, 4, 3),
    ("3連複ワイド 20点（3複全）", 0, 0, 20, 0),
    ("3連単20点（効率ピーク）", 0, 0, 0, 20),
    ("積極 3連単60点（総額ピーク）", 0, 0, 0, 60),
    ("万舟全張り 3連単120点", 1, 0, 0, 120),
]

HISTORICAL = {
    "universe": "妙味（本命≠1号）・クリーン年 3,612レース・1点100円・確定払戻",
    "rows": [
        {"name": "堅実 4点（単1+2連3）", "pts": 4, "roi": 173, "pl": 1049750, "hit": 57, "manshu": 0, "max": 53310},
        {"name": "標準 11点（+3複4+3単3）", "pts": 11, "roi": 302, "pl": 8032610, "hit": 82, "manshu": 118, "max": 231050},
        {"name": "3連複ワイド 20点（3複全）", "pts": 20, "roi": 92, "pl": -585040, "hit": 100, "manshu": 0, "max": 98880},
        {"name": "3連単20点（効率ピーク）", "pts": 20, "roi": 319, "pl": 15847290, "hit": 67, "manshu": 448, "max": 360060},
        {"name": "積極 3連単60点（総額ピーク）", "pts": 60, "roi": 180, "pl": 17250020, "hit": 95, "manshu": 798, "max": 424180},
        {"name": "万舟全張り 3連単120点", "pts": 121, "roi": 99, "pl": -236800, "hit": 100, "manshu": 905, "max": 424180},
    ],
    "note": "3連単20点that効率(回収率)ピーク。総額は60点that最大だが巨額資金＋高分散。"
            "全張り系(3複20/3単120)は的中ほぼ100%でも控除線で負け＝当ててるのに負ける。",
}


def compute(pick: dict, res: dict) -> list[dict]:
    """1レースの pick(買い目/全ランク) と res(結果) から各スタイル収支。
    ワイド系は pick に全ランク(trio_rank/trifecta_rank)が必要。無ければ computable=False。"""
    trio_r = pick.get("trio_rank") or pick.get("trio4") or []
    tf_r = pick.get("trifecta_rank") or pick.get("trifecta3") or []
    ex_r = pick.get("exacta3") or []
    out = []
    for name, ta, ex, tr, tf in STYLES:
        stake = (ta + ex + tr + tf) * 100
        ret = 0
        if ta and res.get("winner") == pick.get("honmei"):
            ret += res.get("tansho_yen") or 0
        if ex and res.get("exacta_combo") in ex_r[:ex]:
            ret += res.get("exacta_yen") or 0
        if tr and res.get("trio_combo") in trio_r[:tr]:
            ret += res.get("trio_yen") or 0
        if tf and res.get("trifecta_combo") in tf_r[:tf]:
            ret += res.get("trifecta_yen") or 0
        computable = (tr <= len(trio_r)) and (tf <= len(tf_r))
        out.append({"name": name, "stake": stake, "ret": ret,
                    "pl": ret - stake, "computable": computable})
    return out


def daily(settled: list[dict]) -> dict:
    """前向き台帳(settled) → 日付別の各スタイル収支。res_full(確定結果)を持つ行のみ集計。

    各行は pick そのもの(honmei/exacta3/trio4/trifecta3/trio_rank/trifecta_rank)＋res_full。
    戻り: {date: {style_name: {"stake","ret","comp"}}}。compはワイド系that全ランクで計算できたか。
    """
    days: dict = {}
    for r in settled:
        res = r.get("res_full")
        if not res:
            continue
        d = r.get("date")
        day = days.setdefault(d, {name: {"stake": 0, "ret": 0, "comp": True}
                                  for name, *_ in STYLES})
        for s in compute(r, res):
            cell = day[s["name"]]
            if s["computable"]:
                cell["stake"] += s["stake"]
                cell["ret"] += s["ret"]
            else:
                cell["comp"] = False
    return days
