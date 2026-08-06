"""買い方くらべ: 賭式×点数の「何点買えば当たるか」。

TRIFECTA_CURVE = 3連単の点数×的中率カーブ（実データ再計算）。
  母集団: モデル本命が1号艇以外（＝イン崩れ）の 36,695レース・6艇成立・確定払戻。
  予測は out-of-sample（当てる年を学習に含めない expanding window）。
  組み合わせの順位付けは Plackett-Luce。

※このモジュールは【的中率】と【万舟捕捉数】だけを持つ。回収率は持たない。
  回収率を名乗るには前向き検証（事前登録→中間境界固定→一発判定）が要る、という
  方針でサイト全体を統一しているため、未検証の回収率は数字ごと置かない。
compute()/daily() は前向き台帳から実収支を出す部分なので従来どおり。
"""
# (表示名, 単勝点, 2連単点, 3連複点, 3連単点)
STYLES = [
    ("堅実 4点（単1+2連3）", 1, 3, 0, 0),
    ("標準 11点（+3複4+3単3）", 1, 3, 4, 3),
    ("3連複ワイド 20点（3複全）", 0, 0, 20, 0),
    ("3連単20点", 0, 0, 0, 20),
    ("積極 3連単60点", 0, 0, 0, 60),
    ("万舟全張り 3連単120点", 1, 0, 0, 120),
]

HISTORICAL = {
    "universe": "モデル本命≠1号艇（イン崩れ）・6艇成立・36,695レース・確定払戻・"
                "out-of-sample予測（当てる年は学習に不使用）・2022〜2026/07",
    "rows": [
        {"name": "堅実 4点（単1+2連3）", "pts": 4, "hit": 38},
        {"name": "3連単 3点", "pts": 3, "hit": 16},
        {"name": "3連単 10点", "pts": 10, "hit": 40},
        {"name": "3連単 20点", "pts": 20, "hit": 59},
        {"name": "3連単 60点", "pts": 60, "hit": 90},
        {"name": "3連複ワイド 20点（3複全）", "pts": 20, "hit": 100},
    ],
    "note": "点数を増やすほど的中率は素直に上がる。ただし増やした点のぶん控除も余計に払うので、"
            "『当たりやすさ』と『効率』は別物。ここでは当たりやすさだけを出す。",
}


# 3連単 点数×的中率カーブ（実データ再計算・out-of-sample）
TRIFECTA_CURVE = {
    "note": "3連単・イン崩れ(モデル本命≠1号)・6艇成立 36,695レース・確定払戻・out-of-sample予測。",
    "points": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 24, 30, 40, 50, 60, 80, 100, 120],
    "hit": [6, 11, 16, 20, 24, 28, 34, 40, 44, 51, 56, 59, 64, 72, 80, 86, 90, 95, 98, 100],
    "manshu": [63, 121, 185, 226, 280, 337, 408, 498, 590, 729, 892, 1020, 1291, 1708, 2523, 3394, 4253, 5404, 6289, 6775],
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
    戻り: {date: {style_name: {"stake","ret","comp"}}}。compはワイド系が全ランクで計算できたか。
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
