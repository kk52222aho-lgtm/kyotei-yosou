"""持続センサー: 「確定払戻で見えたエッジが"未来も通用してるか"」を前向き台帳で継続監視。

背景([[project_kyotei_yosou]]の議論): バックテスト(過去)は忠実な"取れてた額"だが、
  ライブとの差は①パリミュチュエル自己インパクト(薄プールで自分の票that払戻を削る)
  ②過去≠未来(歪みが今年も残ってるか)。①は少額なら小さい。②を測るのがこのセンサー。

思想: エッジの本体は contrarian race selection。低分散の 単勝(≥1.5)/2連単that"持続カナリア"、
  3連複は高分散(fat-tail)の収穫器で informational。前向きの単/2連that 100% を割り"続けたら"
  =歪みが消えたサイン=3連複も道連れ。3連複を直接ROI判定するのはノイズ過多で不可(数百本要る)。

規律(過学習/ノイズ回避):
  - 判定は最小N到達後だけ(単/2連=60, 3連複=120)。それ未満は INSUFFICIENT で保留。
  - fat-tail に配慮し bootstrap CI(レース単位resample)。※裾は過小評価しがち=CIを盲信しない。
  - lifetime と trailing(直近45日)を併記=減衰(最近の失速)を見る。lifetime黒字でも直近が
    崩れてたら WATCH。
  - 見るのは"確定払戻ベースの前向き記録"だけ(log→settle)。過去バックテストは混ぜない。

verdict: INSUFFICIENT(N不足) / HOLDING(点≥100%) / WATCH(点<100%だがCIに100%) / DECAYED(点<100%かつCI上限<100%)

例: python -u -m src.sensor
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

from . import storage
from .papertrade import ODDS_FLOOR

LEDGER = os.path.join(storage.DATA_DIR, "papertrade.jsonl")
MIN_N = {"単勝(≥1.5)": 60, "2連単3点": 60, "3連複4点": 120, "3連単3点": 120}

# センサートラック→動的買い目の賭式キー
TRACK_TO_KEY = {"単勝(≥1.5)": "tansho", "2連単3点": "exacta",
                "3連複4点": "trio", "3連単3点": "trifecta"}
TRAIL_DAYS = 45
BOOT = 2000


def _rows():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _bets(settled, track):
    """トラックごとの (date, stake, ret) をレース単位で返す。"""
    out = []
    for r in settled:
        if track == "単勝(≥1.5)":
            if (r.get("final_odds") or 0) < ODDS_FLOOR:
                continue
            out.append((r["date"], 100, r.get("tansho_return", 0)))
        elif track == "2連単3点":
            pts = r.get("exacta_points", 0)
            if not pts:
                continue
            out.append((r["date"], pts * 100, r.get("exacta_return", 0)))
        elif track == "3連複4点":
            pts = r.get("trio_points", 0)
            if not pts:
                continue
            out.append((r["date"], pts * 100, r.get("trio_return", 0)))
        else:  # 3連単3点
            pts = r.get("trifecta_points", 0)
            if not pts:
                continue
            out.append((r["date"], pts * 100, r.get("trifecta_return", 0)))
    return out


def _stat(bets, seed=0):
    if not bets:
        return None
    stake = np.array([b[1] for b in bets], dtype=float)
    ret = np.array([b[2] for b in bets], dtype=float)
    roi = ret.sum() / stake.sum()
    rng = np.random.default_rng(seed)
    n = len(bets)
    boot = np.empty(BOOT)
    for i in range(BOOT):
        idx = rng.integers(0, n, n)
        boot[i] = ret[idx].sum() / stake[idx].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "roi": roi, "lo": lo, "hi": hi, "pl": int(ret.sum() - stake.sum())}


def _verdict(track, s):
    if s is None or s["n"] < MIN_N[track]:
        have = 0 if s is None else s["n"]
        return "INSUFFICIENT", f"N={have}<{MIN_N[track]} 判定保留（早すぎる結論を機械的に止める）"
    roi, hi = s["roi"] * 100, s["hi"] * 100
    if roi >= 100:
        return "HOLDING", "点≥100% 前向きでも黒字＝歪み生存"
    if hi < 100:
        return "DECAYED", "点<100% かつ CI上限<100% ＝confidently崩れ→撤退検討"
    return "WATCH", "点<100% だがCIに100%含む＝ノイズ/失速の入口→様子見"


def status() -> dict:
    """各トラックの lifetime/trailing 統計と verdict を返す（streamlit等から利用可）。"""
    settled = [r for r in _rows() if r.get("settled")]
    if not settled:
        return {"tracks": {}, "asof": None, "n_settled": 0}
    asof = max(r["date"] for r in settled)
    cut = (dt.datetime.strptime(asof, "%Y%m%d") - dt.timedelta(days=TRAIL_DAYS)).strftime("%Y%m%d")
    out = {"asof": asof, "n_settled": len(settled), "tracks": {}}
    for track in MIN_N:
        allb = _bets(settled, track)
        trailb = [b for b in allb if b[0] >= cut]
        life = _stat(allb, seed=0)
        trail = _stat(trailb, seed=1)
        v, why = _verdict(track, life)
        out["tracks"][track] = {"life": life, "trail": trail, "verdict": v, "why": why}
    return out


def bet_gate() -> dict:
    """動的買い目用ゲート: 各賭式(tansho/exacta/trio/trifecta)を張ってよいか。

    DECAYED(前向きで confidently 100%割れ)の賭式だけ ok=False で自動OFF。
    INSUFFICIENT/HOLDING/WATCH は ok=True（"疑わしきは罰せず"＝崩れ確定まで張る）。
    ＝ライブ実エッジの劣化が、次の買い目に自動反映される（ループを閉じる）。
    """
    tracks = status().get("tracks", {})
    out = {}
    for tname, key in TRACK_TO_KEY.items():
        t = tracks.get(tname)
        v = t["verdict"] if t else "INSUFFICIENT"
        out[key] = {"ok": v != "DECAYED", "verdict": v,
                    "track": tname, "why": (t["why"] if t else "記録なし")}
    return out


def main():
    s = status()
    print("=== 持続センサー（前向き台帳・確定払戻ベース）===")
    if not s["n_settled"]:
        print("  精算済み記録がまだ無い。log→settle が回り出してから。")
        return
    print(f"  as-of {s['asof']} / 精算済 {s['n_settled']} レース / trailing={TRAIL_DAYS}日\n")
    print(f"  {'トラック':<12}{'N':>4}{'ROI':>8}{'CI(95%)':>16}{'直近ROI':>9}  判定")
    for track, t in s["tracks"].items():
        life, trail = t["life"], t["trail"]
        if life is None:
            print(f"  {track:<12}{0:>4}{'—':>8}{'—':>16}{'—':>9}  INSUFFICIENT")
            continue
        ci = f"[{life['lo']*100:.0f},{life['hi']*100:.0f}]%"
        tr = f"{trail['roi']*100:.0f}%({trail['n']})" if trail else "—"
        print(f"  {track:<12}{life['n']:>4}{life['roi']*100:>7.0f}%{ci:>16}{tr:>9}  {t['verdict']}")
    print()
    for track, t in s["tracks"].items():
        print(f"  ・{track}: {t['verdict']} — {t['why']}")
    print("\n  ※判定は低分散の 単勝/2連単that本命(カナリア)。3連複はfat-tailで参考。")
    print("  ※単/2連が前向きで confidently 100%割れ(DECAYED)＝歪み消滅サイン＝3連複も道連れ。")
    print("  ※bootstrap CIは裾を過小評価しがち＝盲信せず、trailingの失速と併せて読む。")


if __name__ == "__main__":
    main()
