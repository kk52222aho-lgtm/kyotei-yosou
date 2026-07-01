@echo off
REM Daily paper-trade run (ASCII only; Japanese in .bat breaks Task Scheduler cmd)
REM Order: settle+push FIRST (result fetching = reliable/quick), THEN scan+log (fragile 3-min scan last)
REM so that even if the scan is interrupted, results are already fetched and pushed.
set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
cd /d C:\dev\kyotei-yosou

echo ==== %date% %time% ==== >> data\papertrade_run.log

REM --- 1) fetch results of bets already logged (quick, the core auto need) ---
"%PY%" -m src.papertrade settle >> data\papertrade_run.log 2>&1
"%PY%" -m src.papertrade report >> data\papertrade_run.log 2>&1
git add data\papertrade.jsonl >> data\papertrade_run.log 2>&1
git commit -m "settle" >> data\papertrade_run.log 2>&1
git push >> data\papertrade_run.log 2>&1

REM --- 2) scan today + log today's picks (fragile long scan, done last) ---
"%PY%" -m src.scan >> data\papertrade_run.log 2>&1
"%PY%" -m src.papertrade log >> data\papertrade_run.log 2>&1
git add data\today_picks.json data\papertrade.jsonl >> data\papertrade_run.log 2>&1
git commit -m "scan+log" >> data\papertrade_run.log 2>&1
git push >> data\papertrade_run.log 2>&1
echo. >> data\papertrade_run.log
