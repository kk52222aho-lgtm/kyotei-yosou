"""買い方(配分): 予算を単勝/2連単/3連単に確信で重み付けして割り振る。

全部100円均等は愚か。当てやすい単勝に厚く、夢の3連単は薄く、確信が高い日ほど単勝集中・
荒れる日ほど散らす。※配分で控除(-EV)は+EVにできない=これは「勝ち」の意味that違う:
的中頻度↑(当たる体感)・分散↓(破産回避=耐久)・確信に張る、の3つを最適化する道具。

flavor: 'safe'=単勝厚め(手堅く) / 'balanced'=標準 / 'dream'=3連単厚め(一発)。
"""
from __future__ import annotations


def _r100(x):
    """100円単位に丸め(最低0)。"""
    return max(int(round(x / 100.0)) * 100, 0)


def _split(combos, probs, pool):
    """poolを買い目に確率比で配分し100円単位に丸め。0円は落とす。"""
    combos = combos or []
    probs = probs or []
    if not combos or pool < 100:
        return []
    ps = [max(float(p), 0.0001) for p in (probs + [0] * len(combos))[:len(combos)]]
    s = sum(ps)
    out = []
    for c, p in zip(combos, ps):
        y = _r100(pool * p / s)
        if y >= 100:
            out.append({"combo": c, "yen": y})
    return out


def plan(pick, budget=1000, flavor="balanced"):
    """pick(win_pct/tansho/exacta3(_p)/trifecta3(_p))と予算→配分。
    返り: {tansho:{target,yen}, exacta:[..], trifecta:[..], total, ratio, note}。"""
    c = min(max((pick.get("win_pct") or 40) / 100.0, 0.20), 0.90)   # 本命の確信
    tilt = {"safe": 0.12, "balanced": 0.0, "dream": -0.15}.get(flavor, 0.0)
    # 単勝=当たりを拾う錨。確信で 0.46→0.66 の狭い帯だけ動かす(極端にしない)
    w_tan = min(max(0.45 + 0.25 * (c - 0.3) + tilt, 0.38), 0.66)
    rem = 1.0 - w_tan
    # 3連単=夢。最低10%は必ず残す(上ブレの芽を消さない)、荒れ/dreamほど厚く最大32%
    w_tri = min(max(0.12 + (0.60 - c) * 0.25 - tilt * 1.3, 0.10), min(0.32, rem - 0.10))
    w_ex = rem - w_tri                                              # 2連単=常に主力(最低10%残る)

    tan = {"target": pick.get("tansho"), "yen": _r100(budget * w_tan)}
    ex = _split(pick.get("exacta3"), pick.get("exacta3_p"), budget * w_ex)
    tri = _split(pick.get("trifecta3"), pick.get("trifecta3_p"), budget * w_tri)

    # 丸め残差は単勝で吸収(=総額を予算にそろえる)
    spent = tan["yen"] + sum(e["yen"] for e in ex) + sum(t["yen"] for t in tri)
    tan["yen"] = max(tan["yen"] + (budget - spent), 0)
    total = tan["yen"] + sum(e["yen"] for e in ex) + sum(t["yen"] for t in tri)

    def pct(y):
        return round(y / total * 100) if total else 0
    ratio = (f"単勝{pct(tan['yen'])}% / 2連単{pct(sum(e['yen'] for e in ex))}% / "
             f"3連単{pct(sum(t['yen'] for t in tri))}%")
    note = ("堅い→単勝を錨に上も少し" if c >= 0.6 else "荒れ→散らして夢を厚く") + f"（確信{c*100:.0f}%・{flavor}）"
    return {"tansho": tan, "exacta": ex, "trifecta": tri, "total": total,
            "ratio": ratio, "note": note}


if __name__ == "__main__":
    demo = [
        ("near-lock堅い", {"win_pct": 82, "tansho": 1, "exacta3": ["1-3", "1-4", "1-5"],
                          "exacta3_p": [0.18, 0.10, 0.07],
                          "trifecta3": ["1-3-4", "1-3-5", "1-4-5"],
                          "trifecta3_p": [0.05, 0.04, 0.03]}),
        ("中間", {"win_pct": 50, "tansho": 2, "exacta3": ["2-1", "2-3", "1-2"],
                "exacta3_p": [0.14, 0.09, 0.08],
                "trifecta3": ["2-1-3", "2-1-5", "2-3-1"],
                "trifecta3_p": [0.045, 0.035, 0.03]}),
        ("荒れ", {"win_pct": 33, "tansho": 4, "exacta3": ["4-1", "4-2", "4-3"],
                "exacta3_p": [0.09, 0.07, 0.06],
                "trifecta3": ["4-1-2", "4-1-3", "4-2-3"],
                "trifecta3_p": [0.03, 0.025, 0.02]}),
    ]
    for name, pk in demo:
        p = plan(pk, 1000)
        print(f"\n[{name}] {p['note']}  → {p['ratio']}  計{p['total']}円")
        print(f"  単勝 {p['tansho']['target']}号: {p['tansho']['yen']}円")
        print(f"  2連単: " + " / ".join(f"{e['combo']}:{e['yen']}円" for e in p['exacta']))
        print(f"  3連単: " + " / ".join(f"{t['combo']}:{t['yen']}円" for t in p['trifecta']))
