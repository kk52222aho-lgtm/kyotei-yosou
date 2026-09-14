"""常時起動ループ: レース時間帯だけ毎分 snapshot() を回す前向きオッズ収集ランナー。

自宅PC/VPSで起動しっぱなしにする(締切T-3/-1分の精度が要るのでGH Actionsのcron遅延では×)。
    起動:  python -m src.odds_probe_loop
    停止:  Ctrl+C
サーバ配慮: モデルは1回だけロード、締切表は30分キャッシュ、窓に入ったレースだけ実フェッチ、
アイドル分は締切表の突き合わせのみ。稼働時間外(深夜)は長めに眠る。賭けは一切しない=記録のみ。
"""
from __future__ import annotations

import datetime as dt
import os
import time
import traceback

from . import odds_probe, predict, export_odds, heartbeat

START_H, END_H = 9, 23        # JST この時間帯だけ稼働(第1R展示前～最終レース締切後)
INTERVAL = 60                 # 秒: 毎分ポーリング
SCHED_TTL = 30 * 60           # 締切表の再取得間隔(秒)
EXPORT_TTL = 60 * 60          # OneDrive退避の間隔(秒)=毎時
HC_LOOP_ENV = "HC_ODDS_LOOP"  # ループ心拍(稼働時間帯毎サイクル)=「生きてる」の沈黙検知


def _ping_loop():
    """ループ心拍。export後pingとは別check＝「生きてる」を独立に証明。

    🚨 2026-09-02: 旧実装は「未設定なら黙って return」やった。壊さん設計としては
    正しいが、**deadman に適用すると deadman 自身が沈黙する**。3回連続で同じ全停止を
    許した原因や。heartbeat.ping は未設定なら10分に1回ログへ吠える。
    """
    heartbeat.ping(heartbeat.LOOP)


def main():
    print(f"odds_probe_loop 起動 {dt.datetime.now():%Y-%m-%d %H:%M}。稼働{START_H}-{END_H}時・Ctrl+Cで停止。")
    heartbeat.announce()                   # 心拍の配線状態を起動ログへ大書き
    bundle = predict.load_model()          # 1回だけ
    sched = None
    sched_date = None
    sched_at = 0.0
    export_at = 0.0
    while True:
        now = dt.datetime.now()
        if not (START_H <= now.hour < END_H):
            time.sleep(300)                # 時間外は5分眠る
            continue
        # 毎時OneDrive退避(バックアップ+export後pingで「保存されてる」を証明)
        if (time.monotonic() - export_at) > EXPORT_TTL:
            try:
                s = export_odds.export(verbose=False)
                export_at = time.monotonic()
                print(f"{now:%H:%M} export OneDrive 有効N={s['effective_n']}/300 "
                      f"両端率={s['both_ends_rate']:.0%}")
            except Exception:
                print(f"{now:%H:%M} exportエラー(継続・pingは出ない=沈黙検知):")
                traceback.print_exc()
        date = now.strftime("%Y%m%d")
        mono = time.monotonic()
        if sched is None or sched_date != date or (mono - sched_at) > SCHED_TTL:
            try:
                sched = odds_probe.fetch_schedule(date)
                sched_date, sched_at = date, mono
                print(f"{now:%H:%M} 締切表更新 {len(sched)}レース")
            except Exception:
                print(f"{now:%H:%M} 締切表取得エラー(継続):")
                traceback.print_exc()
                time.sleep(INTERVAL)
                continue
        try:
            n = odds_probe.snapshot(date, now, bundle=bundle, schedule=sched)
            if n:
                print(f"{now:%H:%M} snapshot {n}行 記録")
        except Exception:
            print(f"{now:%H:%M} snapshotエラー(継続):")
            traceback.print_exc()
        _ping_loop()                       # 稼働時間帯の毎サイクル心拍=「生きてる」の沈黙検知
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
