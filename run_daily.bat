@echo off
REM 競艇予想 日次ペーパートレード運用
REM  1) 確定済みレースを精算  2) 本日の妙味レースを記録(締切前)  3) 実トラック表示
REM Windowsタスクスケジューラで毎朝(例 9:00)実行する想定。

setlocal
set PY=C:\Users\kk522\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe
set PYTHONIOENCODING=utf-8
cd /d C:\dev\kyotei-yosou

echo ==== %date% %time% ====>> data\papertrade_run.log
REM 1) サイト表示用キャッシュを本日ぶんに更新（/today・Streamlitが読む today_picks.json）
"%PY%" -m src.scan               >> data\papertrade_run.log 2>&1
REM 2) ペーパートレード: 前日精算 → 本日記録 → 実トラック
"%PY%" -m src.papertrade settle  >> data\papertrade_run.log 2>&1
"%PY%" -m src.papertrade log     >> data\papertrade_run.log 2>&1
"%PY%" -m src.papertrade report  >> data\papertrade_run.log 2>&1

REM 3) クラウド(Streamlit Cloud)反映: 本日データを GitHub に push（要・git認証）
git add data\today_picks.json data\papertrade.jsonl >> data\papertrade_run.log 2>&1
git commit -m "daily update %date%" >> data\papertrade_run.log 2>&1
git push >> data\papertrade_run.log 2>&1

echo.>> data\papertrade_run.log
endlocal
