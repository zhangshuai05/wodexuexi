$ErrorActionPreference = 'Stop'
$testDir = 'D:\Deepseek_herness\2026-08\diarization'
$target = Join-Path $testDir 'claude_a2.py'
$expected = 'EBD5B41F05C8E353E2759B87DE4E70AA4FB51FA6B9FE32FA0538D84B8B94A277'
if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $expected) {
    throw 'claude_a2.py changed since inspection; inspect the new changes before deployment.'
}
$listeners = @(Get-NetTCPConnection -LocalPort 8734 -State Listen -ErrorAction SilentlyContinue)
$serverIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
if ($serverIds.Count -ne 1) { throw 'Expected one existing server on port 8734.' }
$serverInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($serverIds[0])"
if ($serverInfo.CommandLine -notmatch 'web_server\.py') { throw 'Port 8734 is not the expected web server.' }
$active = @(Get-NetTCPConnection -LocalPort 8734 -State Established -ErrorAction SilentlyContinue)
if ($active.Count -gt 0) { throw 'Server has active connections; retry when inference has finished.' }
$backup = Join-Path $testDir 'claude_a2.py.bak_before_v4_20260908'
if (Test-Path -LiteralPath $backup) { throw 'Backup already exists; inspect it before retrying.' }
Copy-Item -LiteralPath $target -Destination $backup
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'claude_a2.py') -Destination $target -Force
Stop-Process -Id $serverIds[0]
$process = Start-Process -FilePath $serverInfo.ExecutablePath -ArgumentList '-u','web_server.py' -WorkingDirectory $testDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'diar_result\c1_server_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'diar_result\c1_server_stderr.log')
Write-Output "Deployed c1 v4; server PID $($process.Id); backup $backup"
