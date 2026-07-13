"""万舟狙い＝波乱含みレースの『荒れそう度』スコア（参考・エンタメ用／儲け主張なし）。

思想: 観察可能なレース前情報だけで、透明に"荒れやすさ"を採点し理由も出す。
  ※これは分散(荒れやすさ)の指標であって期待値(儲かる)ではない。荒れ条件は公開情報で
    市場that織り込む→当てても勝てるとは限らん([[insight_adversarial_domains]]・test_trio_wide・
    地合いablationで確認済み)。見て楽しむ/参考用。だが"荒れを捉えてるか"は test_wild で裏取り。

スコア成分(各0-1→加重100点):
  anti_inside : 1 - モデルの1号勝率  (モデルthaインを疑うほど高い＝最重視)
  inside_weak : 1号艇の級(B級ほど弱い＝荒れやすい)
  congestion  : 勝率分布のエントロピー(混戦=割れてるほど高い)
  outside_pw  : アウト(4-6号)に強い級がいるか(まくり一撃)
入力 rows: [{lane, win_pct(0-100), racer_class}] （predict_entries出力 or backtestの再構成）。
"""
from __future__ import annotations

import json
import math
import os

from . import storage
from .features import CLASS_MAP

WILD_CACHE = os.path.join(storage.DATA_DIR, "today_wild.json")
WILD_LOG = os.path.join(storage.DATA_DIR, "wild_log.jsonl")   # 荒れ予想の結果検証ログ(前向き累積)
MANSHU = 10000

# class_num(A1=4..B2=1) → 弱さ / 強さ
_WEAK = {4: 0.0, 3: 0.3, 2: 0.7, 1: 1.0, 0: 0.5}
_STRONG = {4: 1.0, 3: 0.6, 2: 0.2, 1: 0.0, 0: 0.1}
_W = {"anti": 0.40, "weak": 0.25, "cong": 0.20, "out": 0.15}


def _cnum(c):
    return CLASS_MAP.get(str(c).strip(), 0)


def score_race(rows: list[dict]) -> dict:
    """rows→ {score(0-100), reasons[list]}。lane/win_pct/racer_class を使う。"""
    n = len(rows) or 1
    p = {int(r["lane"]): float(r.get("win_pct", 100 / n)) / 100 for r in rows}
    p1 = p.get(1, 0.0)
    anti = 1 - p1
    row1 = next((r for r in rows if int(r["lane"]) == 1), None)
    weak = _WEAK.get(_cnum(row1.get("racer_class")) if row1 else 0, 0.5)
    ent = (-sum(v * math.log(v) for v in p.values() if v > 0) / math.log(n)) if n > 1 else 0.0
    out_rows = [r for r in rows if int(r["lane"]) >= 4]
    outp = max((_STRONG.get(_cnum(r.get("racer_class")), 0) for r in out_rows), default=0.0)
    score = 100 * (_W["anti"] * anti + _W["weak"] * weak + _W["cong"] * ent + _W["out"] * outp)

    reasons = []
    if anti >= 0.55:
        reasons.append(f"モデルthaイン軽視(1号{p1*100:.0f}%)")
    if weak >= 0.6 and row1:
        reasons.append(f"1号that{str(row1.get('racer_class', '')).strip() or '格下'}")
    if ent >= 0.85:
        reasons.append("混戦(勝率that割れてる)")
    if outp >= 0.6 and out_rows:
        ol = max(out_rows, key=lambda r: _STRONG.get(_cnum(r.get("racer_class")), 0))
        reasons.append(f"アウトに{str(ol.get('racer_class', '')).strip()}({int(ol['lane'])}号)")
    return {"score": round(score, 1), "reasons": reasons}


def save(date: str, races: list[dict], top: int = 12) -> None:
    """波乱度上位topレースを today_wild.json に保存＋結果検証ログ(未精算)へ追記。"""
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    ranked = sorted(races, key=lambda r: -r["score"])[:top]
    with open(WILD_CACHE, "w", encoding="utf-8") as f:
        json.dump({"date": date, "races": ranked}, f, ensure_ascii=False, indent=1)
    # 結果検証ログに追記（(date,jcd,rno)で重複排除・未精算で置く）
    log = _load_log()
    seen = {(r["date"], r["jcd"], r["rno"]) for r in log}
    for r in ranked:
        if (date, r["jcd"], r["rno"]) in seen:
            continue
        log.append({"date": date, "jcd": r["jcd"], "rno": r["rno"], "venue": r["venue"],
                    "score": r["score"], "honmei": r["honmei"], "settled": False})
    _save_log(log)


def load() -> dict | None:
    if not os.path.exists(WILD_CACHE):
        return None
    with open(WILD_CACHE, encoding="utf-8") as f:
        return json.load(f)


def _load_log() -> list[dict]:
    if not os.path.exists(WILD_LOG):
        return []
    with open(WILD_LOG, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _save_log(rows: list[dict]) -> None:
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    with open(WILD_LOG, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def settle_log(today: str) -> int:
    """未精算の荒れ予想レースの結果を取得＝『ほんとに荒れたか』を刻む（1号飛び/万舟）。"""
    from . import scraper
    log = _load_log()
    n = 0
    for r in log:
        if r.get("settled") or r["date"] > today:
            continue
        res = scraper.fetch_result_full(r["date"], r["jcd"], r["rno"])
        if not res or res.get("winner") is None:
            continue
        r["winner"] = res["winner"]
        r["broke"] = res["winner"] != 1                      # 1号が飛んだ＝荒れた
        r["trifecta_yen"] = res.get("trifecta_yen")
        r["manshu"] = bool(res.get("trifecta_yen") and res["trifecta_yen"] >= MANSHU)
        # 「買えたのか」用: モデル本命の単勝で買ってたら（勝ち=本命なら払戻tansho_yen）
        r["tan_win"] = res["winner"] == r.get("honmei")
        r["tan_return"] = res.get("tansho_yen") if r["tan_win"] else 0
        r["settled"] = True
        n += 1
    _save_log(log)
    return n


def log_stats() -> dict | None:
    """精算済み荒れ予想の集計：本当に荒れた率(1号飛び)・万舟率・平均3連単配当。"""
    settled = [r for r in _load_log() if r.get("settled")]
    if not settled:
        return None
    n = len(settled)
    ys = [r["trifecta_yen"] for r in settled if r.get("trifecta_yen")]

    def tan_pl(grp):
        m = len(grp)
        ret = sum(r.get("tan_return") or 0 for r in grp)
        hit = sum(1 for r in grp if r.get("tan_win"))
        return {"n": m, "hit": hit / m if m else 0, "ret": ret,
                "roi": ret / (m * 100) if m else 0, "pl": ret - m * 100}

    contra = [r for r in settled if r.get("honmei") != 1]   # 実際に張る妙味(本命≠1号)
    return {
        "n": n,
        "broke": sum(1 for r in settled if r.get("broke")) / n,
        "manshu": sum(1 for r in settled if r.get("manshu")) / n,
        "avg_yen": (sum(ys) / len(ys)) if ys else 0,
        "tan_all": tan_pl(settled),                          # 全フラグを本命単勝で買ったら
        "tan_contra": tan_pl(contra) if contra else None,    # 本命≠1号だけ(＝妙味)
        "recent": sorted(settled, key=lambda r: (r["date"], r["jcd"], r["rno"]), reverse=True)[:20],
    }


def main():
    import datetime as dt
    import sys
    today = dt.date.today().strftime("%Y%m%d")
    if len(sys.argv) > 1 and sys.argv[1] == "settle":
        print(f"{settle_log(today)}件の荒れ予想レースを結果精算")
    else:
        s = log_stats()
        print(s if s else "荒れ予想の精算記録なし")


if __name__ == "__main__":
    main()
