"""前向きオッズprobeの採点台=「勝ちの一行」that出る唯一の場所。

収集器(odds_probe)が貯めた odds_timeseries に、事前登録R1/R2を当てて前向き成績を出す:
各ルールの N・的中・単勝ROI・CI・事前登録の判定表(合格/kill/継続)。
late窓(T-1優先→無ければT-3)で判定=暫定オッズ回避(初日知見)。settleは結果を実フェッチ。
racer_id(reg)クラスタbootstrap CI(reg欠測の旧行は素bootstrapにフォールバック)。

例: python -u -m src.odds_probe_score            # 全収集日
    python -u -m src.odds_probe_score 20260717   # 指定日のみ
"""
from __future__ import annotations

import sys
import numpy as np

from . import storage, scraper

N_TARGET = 200                       # 事前登録: 合格/killの母数
LATE = [1, 3]                        # 判定窓(T-1優先, 無ければT-3)


def _decision_rows(conn, date):
    """各レースの判定窓(T-1→T-3)の全艇行を返す {(jcd,rno): {lane: row}}。"""
    out = {}
    races = conn.execute("SELECT DISTINCT jcd,rno FROM odds_timeseries WHERE date=?", (date,)).fetchall()
    for jcd, rno in races:
        rows = None
        for w in LATE:
            r = conn.execute("SELECT combo,odds,model_p,ev,honmei,gyaku,reg,racer_class,nat_win "
                             "FROM odds_timeseries "
                             "WHERE date=? AND jcd=? AND rno=? AND mins_to_deadline=? AND bet_type='tansho'",
                             (date, jcd, rno, w)).fetchall()
            if r:
                rows = {int(x[0]): x for x in r}
                break
        if rows:
            out[(jcd, rno)] = rows
    return out


def fired(conn, date):
    """事前登録R1/R2that発火した賭けを返す [{rule,jcd,rno,target,odds,ev,reg}]。"""
    bets = []
    for (jcd, rno), lanes in _decision_rows(conn, date).items():
        any_row = next(iter(lanes.values()))
        honmei, gyaku = int(any_row[4]), int(any_row[5])
        # R1(大衆過剰反応): 本命=1号 & P(1号)≥.55 & 単勝EV(1号)≥1.05
        if honmei == 1 and 1 in lanes:
            _, od, p, ev, _, _, reg = lanes[1][:7]
            if p is not None and p >= 0.55 and ev is not None and ev >= 1.05:
                bets.append({"rule": "R1", "date": date, "jcd": jcd, "rno": rno, "target": 1,
                             "odds": od, "ev": ev, "reg": reg})
        # R2(逆イン時系列): 本命≠1号 & 単勝EV(本命)≥1.10
        if gyaku == 1 and honmei in lanes:
            _, od, p, ev, _, _, reg = lanes[honmei][:7]
            if ev is not None and ev >= 1.10:
                bets.append({"rule": "R2", "date": date, "jcd": jcd, "rno": rno, "target": honmei,
                             "odds": od, "ev": ev, "reg": reg})
        # R3(純EVフィルタ=控除超える目を"つくる"): 艇を問わず late窓EVが最大&≥1.15 の1艇だけ買う。
        # 順位でなく"価格that割に合う瞬間"だけ拾う=+EVを値から構築する唯一の手。
        best = max(lanes, key=lambda ln: (lanes[ln][3] or 0))
        _, od, p, ev, _, _, reg = lanes[best][:7]
        if ev is not None and ev >= 1.15:
            bets.append({"rule": "R3", "date": date, "jcd": jcd, "rno": rno, "target": best,
                         "odds": od, "ev": ev, "reg": reg})
        # R4(トーナメント発=B1級インの構造妙味): 1号がB1級&全国勝率>5.5 の単勝1点。
        # dev+holdout+機構that揃った唯一の筋(strictは未達)。前向きで確定させる。
        if 1 in lanes and len(lanes[1]) >= 9:
            _, od1, p1, ev1, _, _, reg1, rcls, natw = lanes[1][:9]
            if rcls == "B1" and natw is not None and natw > 5.5:
                bets.append({"rule": "R4", "date": date, "jcd": jcd, "rno": rno, "target": 1,
                             "odds": od1, "ev": ev1, "reg": reg1})
    return bets


def settle(bets):
    """結果を実フェッチして単勝を精算(各betの日付で)。未確定はスキップ。"""
    cache = {}
    out = []
    for b in bets:
        key = (b["date"], b["jcd"], b["rno"])
        if key not in cache:
            cache[key] = scraper.fetch_result_full(b["date"], b["jcd"], b["rno"])
        res = cache[key]
        if not res or not res.get("winner"):
            continue
        won = int(res["winner"]) == b["target"]
        ret = float(res.get("tansho_yen") or 0) if won else 0.0
        out.append({**b, "won": int(won), "ret": ret})
    return out


def cluster_ci(rows, iters=2000, pct=95):
    if not rows:
        return (0, 0)
    ret = np.array([r["ret"] for r in rows], float)
    regs = [r.get("reg") or f"_{i}" for i, r in enumerate(rows)]   # reg欠測は個別=素bootstrap相当
    uniq = list(dict.fromkeys(regs))
    idx = {u: [i for i, g in enumerate(regs) if g == u] for u in uniq}
    rng = np.random.default_rng(11)
    rois = []
    for _ in range(iters):
        pick = [uniq[i] for i in rng.integers(0, len(uniq), len(uniq))]
        sel = [i for u in pick for i in idx[u]]
        rois.append(ret[sel].sum() / (len(sel) * 100) * 100)
    lo = (100 - pct) / 2
    return np.percentile(rois, lo), np.percentile(rois, 100 - lo)


def judge(n, roi, lo, hi):
    if n >= N_TARGET and lo >= 100:
        return "★合格(前向きエッジ確定→③に本物の弾)"
    if (n >= 100 and hi < 100) or (n >= N_TARGET and roi < 90):
        return "×KILL(控除超え無しを確定=帳簿を閉じる)"
    return f"継続(データ蓄積中 N={n}/{N_TARGET}・CIが100跨ぎ)"


def main():
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    conn = storage.connect()
    from . import odds_probe
    odds_probe.ensure_schema(conn)
    dates = ([date_arg] if date_arg else
             [r[0] for r in conn.execute("SELECT DISTINCT date FROM odds_timeseries ORDER BY date").fetchall()])
    all_bets = []
    for d in dates:
        all_bets += fired(conn, d)
    conn.close()
    print(f"収集日 {dates}  発火した賭け(R1/R2) 計{len(all_bets)}件")
    settled = settle(all_bets)
    print("\n=== 前向き成績(=勝ちの一行that出る場所) ===")
    for rule in ["R1", "R2", "R3", "R4"]:
        rows = [s for s in settled if s["rule"] == rule]
        n = len(rows)
        if n == 0:
            print(f"  {rule}: 発火・確定 0件（まだ出てへん）")
            continue
        hit = sum(r["won"] for r in rows)
        roi = sum(r["ret"] for r in rows) / (n * 100) * 100
        lo, hi = cluster_ci(rows)
        print(f"  {rule}: {n}戦 的中{hit}({hit/n*100:.0f}%) ROI{roi:.0f}% "
              f"CI[{lo:.0f},{hi:.0f}]  → {judge(n, roi, lo, hi)}")
    print(f"\n※事前登録: 合格=N≥{N_TARGET}&CI下限≥100 / kill=N≥100&CI上限<100 or N≥{N_TARGET}&ROI<90。")
    print("  late窓(T-1)判定・単勝・確定払戻。オッズ時系列が唯一の未探索フロンティア。")


if __name__ == "__main__":
    main()
