$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$recordings = Join-Path $root "recordings"
$transfer = Join-Path $root "transfer"
$lock = Join-Path $recordings ".recording_active"
if (Test-Path $lock) {
    Write-Error "A recording is active. Stop recording in LANESHIFT before exporting."
    exit 1
}
New-Item -ItemType Directory -Force -Path $transfer | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$zipName = "LANESHIFT-recordings-for-VM-$timestamp.zip"
$zip = Join-Path $transfer $zipName
if (Test-Path $zip) { Remove-Item $zip }
$sessions = Get-ChildItem -Path $recordings -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "session_*" }
if (-not $sessions) {
    Write-Error "No completed recording sessions found in $recordings"
    exit 1
}
Compress-Archive -Path $sessions.FullName -DestinationPath $zip -Force
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
"$hash  $zipName" | Out-File -FilePath (Join-Path $transfer "$zipName.sha256") -Encoding ascii
Write-Host "Created: $zip"
Write-Host "SHA256: $hash"
exit 0
