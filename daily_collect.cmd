@echo off
REM Daily official B/K collection + derived backfills. ASCII ONLY.
REM (Japanese in a .cmd is read as CP932 by cmd.exe and can break the parser so the
REM  task exits 255 every morning while writing nothing - see memory feedback_silent_cron_death.)
REM
REM Why this task exists (2026-09-02): there was NO scheduled task calling src.official.
REM KyoteiOddsExport ran daily, KyoteiEVCapture/KyoteiPaperTrade were Disabled, so
REM entries/payouts silently fell 27 days behind (last 20260806) while the export log
REM kept moving. The envelope could not be judged because 168 of its 357 races had no payout.
REM
REM Order matters: official first (downloads K files), then the parsers that read those files.
REM Window is the last 10 days: official skips already-cached lzh, so re-running is cheap
REM and it self-heals a few missed days without a full backfill.

set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\dev\kyotei-yosou

for /f %%d in ('powershell -NoProfile -Command "(Get-Date).AddDays(-10).ToString('yyyyMMdd')"') do set FROM=%%d
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd')"') do set TO=%%d

"%PY%" -m src.official --start %FROM% --end %TO% >> data\daily_collect.log 2>&1
"%PY%" -m src.backfill_st                        >> data\daily_collect.log 2>&1
"%PY%" -m src.backfill_tenji_st                  >> data\daily_collect.log 2>&1
"%PY%" -m src.heartbeat show                     >> data\daily_collect.log 2>&1

REM 2026-09-08: push the status beacon so GitHub Actions (deadman.yml) can see from
REM outside whether collection is alive. This is the ONLY external observer right now
REM (HC_ODDS_LOOP / HC_ODDS_EXPORT / HEALTHCHECK_DAILY are all still unset).
REM It uses the Contents API with GITHUB_TOKEN from C:\dev\.env and never touches
REM the working tree - this repo carries unrelated uncommitted changes.
REM 2026-09-08: oriten (BOATCAST original exhibition data) had been stale for 54 days
REM because nothing scheduled it. Same hole as everywhere else today: the collector
REM existed and worked, it just was never registered. 1.5s per race, so a 10-day
REM window is a few minutes. Existing races are skipped.
"%PY%" -m src.collect_oriten --start %FROM% --end %TO% >> data\daily_collect.log 2>&1

REM 2026-09-11: model.joblib was 56 days STALE while status_beacon printed it every
REM morning. The beacon was right and nobody read it: nothing ever called src.train.
REM Same hole as src.official (27d) and oriten (54d) - the tool existed, it was just
REM never wired into a loop. refresh_derived rebuilds only what is past its own age
REM limit (model 7d, motor_boat 30d - both shorter than the beacon tolerance so it
REM gets fixed before the beacon complains) and verifies the file actually got newer.
"%PY%" -m src.refresh_derived                    >> data\daily_collect.log 2>&1

"%PY%" -m src.status_beacon push                 >> data\daily_collect.log 2>&1
