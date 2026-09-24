@echo off
REM Daily: fetch the Miyajima paper and record the FROZEN ngram prediction
REM BEFORE the races. ASCII ONLY.
REM (Japanese in a .cmd is read as CP932 by cmd.exe and can break the parser so the
REM  task exits 255 every morning while writing nothing.)
REM
REM Why (2026-09-24): the holdout was looked at twice (dictionary scoring, then a
REM learned char-ngram). Nothing is left but a forward test. See
REM docs/prereg_paper_ngram.md - the model is frozen and its SHA256 is written there.
REM
REM Timing: two triggers, 09:30 and 10:10. Miyajima R1 usually closes after 10:30,
REM so both are before every deadline. INSERT OR IGNORE keeps the FIRST record as the
REM witness. The printed deadline of each race is stored next to recorded_at so that
REM "it was recorded before the deadline" can be checked per race at judgement time.

set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\dev\kyotei-yosou

"%PY%" -m src.paper_daily >> data\paper_daily.log 2>&1
