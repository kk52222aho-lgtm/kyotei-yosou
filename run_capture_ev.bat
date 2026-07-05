@echo off
REM 2rentan EV capture (ASCII only; Japanese in .bat breaks Task Scheduler cmd).
REM Runs every ~15 min. Scrapes exacta closing odds for unsettled picks, computes
REM EV=sum(p*odds)/3, records to ledger (overwrites -> converges to closing value).
REM EV>2.0 is the validated buy band (test_ev_probe: robust 300-500%). Off-hours = no-op.
set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
cd /d C:\dev\kyotei-yosou

echo ==== %date% %time% EV capture ==== >> data\ev_capture.log
"%PY%" -m src.papertrade capture_ev >> data\ev_capture.log 2>&1
git add data\papertrade.jsonl >> data\ev_capture.log 2>&1
git commit -m "capture ev" >> data\ev_capture.log 2>&1
git push >> data\ev_capture.log 2>&1
