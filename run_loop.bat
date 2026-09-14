@echo off
REM Forward odds-dynamics collection loop (freeze experiment). Runs 9-23h, one snapshot/min.
REM ASCII only. -u = unbuffered so probe_loop.log stays live for silence-detection by eye.
REM Real python path (WindowsApps alias does not resolve under Task Scheduler / Startup).
REM 2026-09-08: stdout was cp932 here, so an emoji in heartbeat.announce() raised
REM UnicodeEncodeError and killed the loop at startup. The watchdog then failed to
REM restart it 286 times over 3 days. Belt and braces: force UTF-8 here as well,
REM and keep every print in the daemon path ASCII-only (see src/heartbeat.py).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\dev\kyotei-yosou
"C:\Users\kk522\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe" -u -m src.odds_probe_loop >> data\probe_loop.log 2>&1
