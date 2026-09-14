# Kyotei odds-probe loop watchdog. ASCII ONLY (PS 5.1 parses non-ASCII in BOM-less
# .ps1 as CP932 and can die silently with exit 0 - see memory insight_ps51_silent_exit_zero).
#
# Why this exists: the loop died 2026-07-31 (found 08-03) and again 2026-08-19
# (found 09-02, by hand). The Startup-folder .bat only fires at logon, so a mid-day
# Ctrl+C or crash leaves it dead for weeks. This runs every 10 minutes and restarts it.
# It writes a line EVERY run (alive or not) so the log's mtime is itself a liveness signal.

$ErrorActionPreference = 'Continue'
$root = 'C:\dev\kyotei-yosou'
$log  = Join-Path $root 'data\watchdog.log'
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'

$hour = (Get-Date).Hour
if ($hour -lt 9 -or $hour -ge 23) {
    Add-Content -Path $log -Value "$stamp off-hours" -Encoding utf8
    exit 0
}

# Match on CommandLine, but ONLY for python processes. Two traps here, both hit on 2026-09-02:
#   1) the real process is python3.12.exe, NOT python.exe - a Name='python.exe' filter
#      never matches, so the watchdog would relaunch every 10 min and stack duplicates
#      (two loops ran simultaneously in July 2026 for exactly this class of reason)
#   2) the querying powershell.exe has 'odds_probe_loop' in its OWN command line and
#      matches itself - so it must be excluded by name
function Get-LoopProcs {
    Get-CimInstance Win32_Process -ErrorAction Stop |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*odds_probe_loop*' }
}

$p = $null
try { $p = @(Get-LoopProcs) } catch {
    Add-Content -Path $log -Value "$stamp process-query-failed: $($_.Exception.Message)" -Encoding utf8
    exit 1
}

if ($p.Count -gt 1) {
    $ids = ($p | ForEach-Object { $_.ProcessId }) -join ','
    Add-Content -Path $log -Value "$stamp DUPLICATE LOOPS pid=$ids - fix by hand" -Encoding utf8
    exit 1
}
if ($p.Count -eq 1) {
    Add-Content -Path $log -Value "$stamp alive pid=$($p[0].ProcessId)" -Encoding utf8
    exit 0
}

Add-Content -Path $log -Value "$stamp NOT RUNNING - restarting" -Encoding utf8
Start-Process -FilePath (Join-Path $root 'run_loop.bat') -WorkingDirectory $root -WindowStyle Hidden
Start-Sleep -Seconds 8
$q = @(Get-LoopProcs)
if ($q.Count -ge 1) {
    Add-Content -Path $log -Value "$stamp restarted pid=$($q[0].ProcessId)" -Encoding utf8
} else {
    Add-Content -Path $log -Value "$stamp RESTART FAILED" -Encoding utf8
    exit 1
}
