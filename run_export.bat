@echo off
REM odds_timeseries -> OneDrive daily-partition backup (independent of the running loop).
REM ASCII only. Real python path (WindowsApps alias does not resolve under Task Scheduler).
cd /d C:\dev\kyotei-yosou
"C:\Users\kk522\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe" -m src.export_odds >> data\export_task.log 2>&1
