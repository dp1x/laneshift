param(
    [string]$Model = $env:LANESHIFT_ONNX_MODEL,
    [string]$Tokenizer = $env:LANESHIFT_ONNX_TOKENIZER,
    [int]$Port = 8766,
    [int]$Length = 256,
    [int]$Threads = 0,
    [string]$Godot = "godot",
    [string]$ModelId = "",
    [ValidateSet("original_model", "fine_tuned_model")]
    [string]$LabelSource = "original_model",
    [int]$Episodes = 999,
    [double]$EpisodeDuration = 0,
    [switch]$Windowed,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonDir = Join-Path $root "python"
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not $Model) {
    $Model = "C:\Users\Dhane\dev\kev-onnx-backup\kev07b_fp16_L256.onnx"
}
if (-not $Tokenizer) {
    $cache = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B-Base\snapshots"
    $found = Get-ChildItem $cache -Recurse -Filter "tokenizer.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $Tokenizer = $found.FullName }
}
if (-not (Test-Path $Model)) {
    throw "ONNX model not found: $Model. Set LANESHIFT_ONNX_MODEL or pass -Model."
}
if (-not (Test-Path $Tokenizer)) {
    throw "Tokenizer not found: $Tokenizer. Set LANESHIFT_ONNX_TOKENIZER or pass -Tokenizer."
}
if (-not $ModelId) {
    $ModelId = "jaredpalmer/kev-0.8b@v1.0+onnx-fp16:L$Length"
}

$controllerLog = Join-Path $logDir "kev-controller.log"
$controllerErr = Join-Path $logDir "kev-controller.err.log"
Remove-Item $controllerLog, $controllerErr -Force -ErrorAction SilentlyContinue

$controllerArgs = @(
    "server.py",
    "--policy", "onnx_kev",
    "--port", "$Port",
    "--onnx-model", $Model,
    "--onnx-tokenizer", $Tokenizer,
    "--onnx-length", "$Length",
    "--onnx-threads", "$Threads",
    "--onnx-model-id", $ModelId,
    "--onnx-label-source", $LabelSource,
    "--decision-timeout", "10.0",
    "--max-state-age", "10.0"
)

Write-Host "Starting local Kev ONNX controller on port $Port..."
$controller = Start-Process -FilePath "python" -ArgumentList $controllerArgs -WorkingDirectory $pythonDir -PassThru -WindowStyle Hidden -RedirectStandardOutput $controllerLog -RedirectStandardError $controllerErr

try {
    $deadline = (Get-Date).AddSeconds(90)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($controller.HasExited) {
            throw "Controller exited before opening the port. See $controllerErr"
        }
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $client.Connect("127.0.0.1", $Port)
            $ready = $true
        } catch {
        } finally {
            $client.Dispose()
        }
        if ($ready) { break }
        Start-Sleep -Milliseconds 300
    }
    if (-not $ready) {
        throw "Controller did not open 127.0.0.1:$Port. See $controllerErr"
    }

    Write-Host "Controller ready. Launching LANESHIFT with Kev in control."
    Write-Host "Model: $ModelId"
    Write-Host "Logs:  $controllerLog"

    $godotArgs = @()
    if ($Headless) { $godotArgs += "--headless" }
    $godotArgs += @(
        "--path", $root,
        "--",
        "--auto-run",
        "--seeds=1",
        "--episodes=$Episodes",
        "--episode-duration=$EpisodeDuration",
        "--ws-port=$Port",
        "--traffic-density=0.7",
        "--sim-speed-scale=1.0",
        "--external-timeout-ms=12000",
        "--max-stale-ticks=6000"
    )
    if ($Windowed) { $godotArgs += "--windowed" }

    & $Godot @godotArgs
    exit $LASTEXITCODE
} finally {
    if ($controller -and -not $controller.HasExited) {
        Stop-Process -Id $controller.Id -Force -ErrorAction SilentlyContinue
    }
}
