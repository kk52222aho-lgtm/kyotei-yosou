@echo off
REM Daily snapshot of venue official racer-comment / preview pages. ASCII ONLY.
REM (Japanese in a .cmd is read as CP932 by cmd.exe and can break the parser so the
REM  task exits 255 every morning while writing nothing.)
REM
REM Why (2026-09-23): these pages are overwrite-only. targetday is accepted and
REM silently discarded (md5 identical across 4 different dates at 3 venues), and the
REM Internet Archive has 1-2 snapshot days over 4 years. Every day not captured is
REM gone for good. Only 2 venues (21 Ashiya, 23 Karatsu) have index_racers_comment;
REM 10 have index_tenbo. The list was verified by counting page= values on all 24
REM venue front pages - an unknown page= value silently falls back to the index, so
REM HTTP 200 plus plausible body text is NOT proof the page exists.
REM
REM Raw HTML only, gzipped. Parsing comes later (same split as chirashi-otoku).
REM Two runs per day: morning venues publish early, night venues publish late.

set PY=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\dev\kyotei-yosou

"%PY%" -m src.collect_venue_snapshot >> data\venue_snap.log 2>&1
