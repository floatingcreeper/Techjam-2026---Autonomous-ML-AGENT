# Supervisor: keep the agent loop running for a fixed wall-clock window, restarting it if the
# process dies outright.
#
#   .\run_for_hours.ps1                  # 2 hours, default settings
#   .\run_for_hours.ps1 -Hours 6         # 6 hours
#   .\run_for_hours.ps1 -Hours 2 -GitSnapshot
#
# Two different failure levels are already handled WITHOUT this script, and it is worth knowing
# which is which before relying on it:
#
#   1. A candidate that fails (bad generated code, a crash inside train(), an unaffordable
#      config) is caught by the orchestrator, logged as a rejected iteration, and the loop moves
#      on by itself. This is normal operation, not a crash - the supervisor is not involved.
#   2. A transient Ollama failure is retried with backoff inside the loop (LLM_RETRY_DELAYS_S).
#
# This script only covers level 3: the whole PROCESS dying - Ollama down for longer than the
# backoff, an OOM kill, a reboot, a stray Ctrl+C. Because agent/resume.py persists
# {last_completed_iteration, current_best} after every iteration and the solution tree saves
# alongside it, re-running the same command continues from where it stopped rather than starting
# over, so a restart loses at most the one iteration that was in flight.

param(
    [double]$Hours = 2.0,
    [int]$Iterations = 200,
    [string]$DataDir = './KuaiRand-Pure/data',
    [switch]$GitSnapshot,
    [switch]$StopOnConvergence
)

$deadline = (Get-Date).AddHours($Hours)
$logDir = Join-Path $PSScriptRoot 'runs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ('supervisor_{0:yyyyMMdd_HHmmss}.log' -f (Get-Date))
$attempt = 0

Write-Host "[supervisor] running until $deadline (log: $log)"

while ((Get-Date) -lt $deadline) {
    $attempt++
    # Hand the loop the time it has LEFT, so the budget tracks the wall clock across restarts
    # instead of resetting to the full window every time the process comes back.
    $remaining = ($deadline - (Get-Date)).TotalHours
    if ($remaining -le 0.01) { break }

    $agentArgs = @('-u', '-m', 'agent.cli', 'run',
                   '--data_dir', $DataDir,
                   '--iterations', $Iterations,
                   '--max-hours', ([math]::Round($remaining, 4)))
    if (-not $StopOnConvergence) { $agentArgs += '--ignore-convergence' }
    if ($GitSnapshot)            { $agentArgs += '--git-snapshot' }

    Write-Host ("[supervisor] attempt {0}, {1:N2}h left" -f $attempt, $remaining)

    # Start-Process rather than `& python ... 2>&1 | Tee-Object`: in Windows PowerShell 5.1,
    # piping a NATIVE executable's redirected stderr wraps each line in an ErrorRecord
    # (NativeCommandError) and sets $? to false even on a clean exit 0, which would make this
    # supervisor misread every successful run as a crash and restart it forever.
    $out = "$log.attempt$attempt.out"
    $err = "$log.attempt$attempt.err"
    $p = Start-Process -FilePath 'python' -ArgumentList $agentArgs -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $code = $p.ExitCode

    "=== supervisor attempt $attempt at $(Get-Date), exit $code ===" |
        Add-Content -Path $log -Encoding utf8
    foreach ($f in @($out, $err)) {
        if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
            Get-Content $f | Add-Content -Path $log -Encoding utf8
        }
        if (Test-Path $f) { Remove-Item $f -Force }
    }
    Get-Content $log -Tail 12

    if ($code -eq 0) {
        Write-Host "[supervisor] loop exited cleanly (budget reached, cap hit, or converged)"
        break
    }

    Write-Host "[supervisor] process died with exit code $code - resuming from state.json"
    # Brief pause so a hard-down Ollama isn't hammered in a tight restart loop.
    Start-Sleep -Seconds 20
}

Write-Host "[supervisor] finished after $attempt attempt(s). Full log: $log"
Write-Host "[supervisor] run 'python -m agent.cli status' for current-best and cost."
