"""勝てる目=高的中の堅い本命(イン×強い選手 or 高確信)を選ぶ。儲けでなく"当たる"で出す。

今日の確定: 予測構造は本物(1にイン#1・2に地力#2)だがエッジは無い(市場が値付け済)。
∴逆張り妙味(カラクリで死)でなく、当たる本命を出す。test_katai(確定払戻・walk-forward・6艇):
  本命=1号 かつ インが地力1位 → 単勝的中73.7%(鉄板)
  本命=1号 かつ インが地力1-2位 → 68.8%
  モデル確信(正規化勝率)≥50% → 63.3%
回収は89-94%<100=当たるが控除で儲けは出ない(市場が値付け済)。的中率だけで堅軸を出す。
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _power_rank(rows):
    """各艇の地力(全国勝率)順位。1=6人中最強。"""
    nat = {e["lane"]: (e.get("nat_win") or 0.0) for e in rows}
    order = sorted(nat, key=lambda l: -nat[l])
    return {l: i + 1 for i, l in enumerate(order)}


def select(rows):
    """堅い本命なら勝てる目dictを、そうでなければ None を返す。rows=predict_entries出力(rank順)。"""
    if not rows:
        return None
    honmei = rows[0]
    prank = _power_rank(rows)
    in_rank = prank.get(1)                      # イン(1号)の地力順位
    conf = honmei.get("win_pct") or 0.0
    strong_inside = honmei["lane"] == 1 and in_rank is not None and in_rank <= 2
    high_conf = conf >= 50.0
    if not (strong_inside or high_conf):
        return None                             # 堅くない=勝てる目でない

    if honmei["lane"] == 1 and in_rank == 1:
        tier, hit = "◎鉄板", 74                  # イン×地力1位=単勝73.7%
    elif strong_inside:
        tier, hit = "◎堅い", 69                  # イン×地力1-2位=68.8%
    else:
        tier, hit = "○本命堅め", 63              # 高確信本命=63.3%

    # 2連単 本命-流し 上位3(Harville)
    p = {e["lane"]: e["win_prob"] for e in rows}
    s = sum(p.values()) or 1
    p = {k: v / s for k, v in p.items()}
    h = honmei["lane"]
    dh = 1 - p[h]
    ex = sorted(((f"{h}-{j}", p[h] * p[j] / dh if dh > 0 else 0)
                 for j in p if j != h), key=lambda x: -x[1])
    return {
        "tansho": h, "name": honmei.get("name"), "tier": tier, "hit_pct": hit,
        "conf": round(conf, 1), "in_power_rank": in_rank,
        "exacta3": [c for c, _ in ex[:3]],
    }


def save(date, picks):
    DATA.mkdir(exist_ok=True)
    (DATA / "today_katai.json").write_text(
        json.dumps({"date": date, "picks": picks}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def load():
    f = DATA / "today_katai.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))
