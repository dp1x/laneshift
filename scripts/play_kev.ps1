param(
    [string]$Model = $env:LANESHIFT_ONNX_MODEL,
    [string]$Tokenizer = $env:LANESHIFT_ONNX_TOKENIZER,
    [int]$Port = 8766,
    [int]$Length = 256,
    [int]$Threads = 8,
    [string]$Godot = "godot",
    [string]$ModelId = "",
    [ValidateSet("original_model", "fine_tuned_model")]
    [string]$LabelSource = "original_model",
    [ValidateSet("compact", "lane")]
    [string]$PromptMode = "lane",
    [string]$Seeds = "1,2,3,4,5,6,7,8",
    [int]$Episodes = 999,
    [double]$EpisodeDuration = 0,
    [double]$DecisionInterval = 1.0,
    [int]$TurnTimeoutMs = 25000,
    [switch]$Windowed,
    [switch]$Headless,
    [switch]$NoSafety,
    [switch]$Evaluation
)

$ErrorActionPreference = "Stop"
$packaged = Test-Path (Join-Path $PSScriptRoot "LANESHIFT.exe")
if ($packaged) {
    $root = $PSScriptRoot
} else {
    $root = Split-Path -Parent $PSScriptRoot
}
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$cModel = "C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx"
$cTokenizer = "C:\LANESHIFT2D-Kev-model\tokenizer.json"
if ($packaged) {
    $pythonExe = Join-Path $root "runtime\py\python.exe"
    $pythonDir = Join-Path $root "controller"
    $nextExe = Join-Path $root "LANESHIFT-next.exe"
    $gameExe = if (Test-Path $nextExe) { $nextExe } else { Join-Path $root "LANESHIFT.exe" }
    if (-not $Model) {
        $Model = if (Test-Path $cModel) { $cModel } else { Join-Path $root "runtime\kev07b_fp16_L256.onnx" }
    }
    if (-not $Tokenizer) {
        $Tokenizer = if (Test-Path $cTokenizer) { $cTokenizer } else { Join-Path $root "runtime\tokenizer.json" }
    }
} else {
    $pythonExe = "python"
    $pythonDir = Join-Path $root "python"
    $gameExe = ""
    if (-not $Model) {
        $Model = if (Test-Path $cModel) { $cModel } else { "C:\Users\Dhane\dev\kev-onnx-backup\kev07b_fp16_L256.onnx" }
    }
    if (-not $Tokenizer) {
        $Tokenizer = if (Test-Path $cTokenizer) { $cTokenizer } else {
            $cache = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B-Base\snapshots"
            $found = Get-ChildItem $cache -Recurse -Filter "tokenizer.json" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) { $found.FullName }
        }
    }
}
if (-not (Test-Path $Model)) {
    throw "ONNX model not found: $Model. Set LANESHIFT_ONNX_MODEL or pass -Model."
}
if (-not (Test-Path $Tokenizer)) {
    throw "Tokenizer not found: $Tokenizer. Set LANESHIFT_ONNX_TOKENIZER or pass -Tokenizer."
}
if (-not $ModelId) {
    $ModelId = "jaredpalmer/kev-0.8b@v1.0+onnx:L$Length"
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
    "--kev-prompt", $PromptMode,
    "--decision-timeout", $(if ($Evaluation) { "20.0" } else { "10.0" }),
    "--max-state-age", $(if ($Evaluation) { "60.0" } else { "10.0" })
)
if ($Evaluation) {
    $controllerArgs += "--eval-mode"
} elseif (-not $NoSafety) {
    $controllerArgs += "--safety-fallback"
}

Write-Host "Starting local Kev ONNX controller on port $Port..."
$controller = Start-Process -FilePath $pythonExe -ArgumentList $controllerArgs -WorkingDirectory $pythonDir -PassThru -WindowStyle Hidden -RedirectStandardOutput $controllerLog -RedirectStandardError $controllerErr

try {
    $deadline = (Get-Date).AddSeconds(180)
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
    # The TCP port opens a moment before clients are served reliably. Give the
    # event loop and the first inference warm-up a beat before launching.
    Start-Sleep -Milliseconds 1000
    if ($controller.HasExited) {
        throw "Controller exited immediately after opening the port. See $controllerErr"
    }

    Write-Host "Controller ready. Launching LANESHIFT with Kev in control."
    Write-Host "Model: $ModelId"
    Write-Host "Logs:  $controllerLog"

    $gameArgs = @(
        "--auto-run",
        "--seeds=$Seeds",
        "--episodes=$Episodes",
        "--episode-duration=$EpisodeDuration",
        "--ws-port=$Port",
        "--traffic-density=0.7",
        "--sim-speed-scale=1.0",
        "--max-fps=30",
        "--external-timeout-ms=$(if ($Evaluation) { 60000 } else { 12000 })",
        "--max-stale-ticks=6000"
    )
    if ($Evaluation) {
        $gameArgs += @(
            "--turn-based",
            "--eval-mode",
            "--decision-interval=$DecisionInterval",
            "--turn-timeout-ms=$TurnTimeoutMs"
        )
    }
    if ($Windowed) { $gameArgs += "--windowed" }

    if ($packaged) {
        $exeArgs = @()
        if ($Headless) { $exeArgs += "--headless" }
        $exeArgs += "--"
        $exeArgs += $gameArgs
        Write-Host ("Game: " + $gameExe + " " + ($exeArgs -join " "))
        # Use Start-Process for the exported executable. PowerShell's call
        # operator can re-bind native arguments containing '=' and '--'; the
        # game then silently starts with the default port and never connects.
        $gameProcess = Start-Process -FilePath $gameExe -ArgumentList $exeArgs -PassThru -NoNewWindow
        Wait-Process -Id $gameProcess.Id
        exit $gameProcess.ExitCode
    } else {
        $godotArgs = @()
        if ($Headless) { $godotArgs += "--headless" }
        $godotArgs += @("--path", $root, "--") + $gameArgs
        & $Godot @godotArgs
    }
    exit $LASTEXITCODE
} finally {
    if ($controller -and -not $controller.HasExited) {
        Stop-Process -Id $controller.Id -Force -ErrorAction SilentlyContinue
    }
}
