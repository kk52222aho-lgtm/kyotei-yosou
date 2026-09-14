@echo off
REM Daily forward record for the pushout detector. ASCII ONLY.
REM (Japanese in a .cmd is read as CP932 by cmd.exe and can break the parser so the
REM  task exits 255 every morning while writing nothing - memory feedback_silent_cron_death.)
REM
REM Preregistration: docs/prereg_pushout.md (2026-09-03). Model and threshold are FROZEN.
REM Never retrain, never move the threshold. This job only records.
REM
REM Order: settle yesterday first (payouts must already be collected by KyoteiDailyCollect
REM at 07:30), then log today. Logging happens before deadlines, results are never consulted.

set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\dev\kyotei-yosou

"%PY%" -m src.pushout_record settle >> data\pushout_record.log 2>&1
"%PY%" -m src.pushout_record log    >> data\pushout_record.log 2>&1
"%PY%" -m src.pushout_record report >> data\pushout_record.log 2>&1

REM 2026-09-08: R5 "shrink" forward record. Preregistration: docs/prereg_shrink.md.
REM Reads odds_timeseries (already collected by the probe loop) - no extra fetching.
REM N counts from 20260909 only; 20260716-0905 was used for exploration and is excluded.
REM Frozen: window T-10/T-1, group split at move<0, primary endpoint = shrink minus wide.
"%PY%" -m src.shrink_record settle  >> data\shrink_record.log 2>&1
"%PY%" -m src.shrink_record log     >> data\shrink_record.log 2>&1
"%PY%" -m src.shrink_record report  >> data\shrink_record.log 2>&1
