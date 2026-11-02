param(
    [string]$Model = $env:LANESHIFT_ONNX_MODEL,
    [string]$Tokenizer = $env:LANESHIFT_ONNX_TOKENIZER,
    [string]$Data = $env:LANESHIFT_ONNX_DATA,
    [string]$Python = $env:LANESHIFT_PYTHON,
    [string]$Godot = $env:GODOT,
    [int]$Port = 8766,
    [int]$Length = 256,
    [int]$Threads = 4,
    [ValidateSet("cpu", "qnn", "dml")]
    [string]$Provider = "cpu",
    [string]$ModelId = "",
    [ValidateSet("original_model", "fine_tuned_model")]
    [string]$LabelSource = "original_model",
    [ValidateSet("compact", "lane", "scene")]
    [string]$PromptMode = "scene",
    [string]$Seeds = "1,2,3,4,5,6,7,8",
    [int]$Episodes = 0,
    [int]$MaxEpisodes = 0,
    [double]$EpisodeDuration = 0,
    [double]$DecisionInterval = 1.0,
    [double]$MaxDecisionAge = 5.0,
    [int]$RequestTimeoutMs = 25000,
    [int]$InferenceCooldownMs = 250,
    [int]$TurnTimeoutMs = 35000,
    [double]$ControllerTimeout = 0,
    [double]$TrafficDensity = 0.7,
    [int]$MaxFps = 60,
    [switch]$Windowed,
    [switch]$Headless,
    [switch]$NoSafety,
    [switch]$Evaluation,
    [switch]$TurnBased,
    [switch]$Watch,
    [switch]$DryRun
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

if ($Evaluation -or $TurnBased) {
    $mode = "eval"
} elseif ($Watch) {
    $mode = "watch"
} else {
    $mode = "live"
}
if ($Evaluation -and $Watch) {
    throw "Choose either -Evaluation (strict turn-based) or -Watch (legacy safety mode), not both."
}

function Resolve-FirstPath {
    param([string[]]$Candidates, [string]$Label)
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $rendered = ($Candidates | Where-Object { $_ } | ForEach-Object { "  $_" }) -join [Environment]::NewLine
    throw "$Label not found. Looked in:$([Environment]::NewLine)$rendered"
}

function Quote-Arg {
    param([string]$Value)
    if ($Value -match '\s') {
        return '"' + $Value + '"'
    }
    return $Value
}

# --- Python -----------------------------------------------------------------
if ($packaged) {
    $venvPython = Join-Path $root "runtime\py\python.exe"
    $pythonDir = Join-Path $root "controller"
    $nextExe = Join-Path $root "LANESHIFT-next.exe"
    $gameExe = if (Test-Path $nextExe) { $nextExe } else { Join-Path $root "LANESHIFT.exe" }
} else {
    $venvPython = Join-Path $root ".venv-host\Scripts\python.exe"
    $pythonDir = Join-Path $root "python"
    $gameExe = ""
}
if (-not $Python) {
    if (Test-Path -LiteralPath $venvPython) {
        $Python = $venvPython
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $Python = (Get-Command python).Source
    } else {
        throw "Python not found. Run SETUP-HOST.bat, or pass -Python or set LANESHIFT_PYTHON."
    }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

# Native Python wrappers can write benign warnings to stderr. Do not let that
# abort the launcher; parse a machine-readable line and check the exit code.
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$pythonProbe = & $Python -c "import json, platform, sys; import onnxruntime as o; print(json.dumps({'machine': platform.machine(), 'version': sys.version.split()[0], 'ort': o.__version__, 'providers': o.get_available_providers()}))" 2>&1
$pythonExit = $LASTEXITCODE
$moduleProbe = & $Python -c "import websockets, onnxruntime, tokenizers, numpy, httpx; print('LANESHIFT_DEPS_OK')" 2>&1
$moduleExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorAction
if ($pythonExit -ne 0) {
    Write-Host ($pythonProbe -join [Environment]::NewLine)
    throw "Python at $Python cannot import onnxruntime. Run SETUP-HOST.bat or install python\requirements.txt."
}
$probeLines = @($pythonProbe | Where-Object { $_ -ne "" })
$jsonLine = @($probeLines | Where-Object { $_ -match '^\{' } | Select-Object -Last 1)
if (-not $jsonLine) {
    Write-Host ($probeLines -join [Environment]::NewLine)
    throw "Could not read the Python/ONNX Runtime probe output from $Python."
}
$pythonInfo = $jsonLine | ConvertFrom-Json
$pythonMachine = [string]$pythonInfo.machine
$pythonVersion = [string]$pythonInfo.version
$ortVersion = [string]$pythonInfo.ort
$ortProviders = (@($pythonInfo.providers) -join ",")
$processorIdentifier = [string]$env:PROCESSOR_IDENTIFIER
$emulated = ($pythonMachine -eq "AMD64" -or $pythonMachine -eq "x86") -and ($processorIdentifier -match "ARM|Qualcomm")
if ($emulated) {
    Write-Warning "Python is x64 under ARM64 emulation. Inference will be slow. Install native Windows ARM64 Python and run SETUP-HOST.bat for the host baseline."
}

if ($moduleExit -ne 0 -or (($moduleProbe -join "`n") -notmatch "LANESHIFT_DEPS_OK")) {
    Write-Host ($moduleProbe -join [Environment]::NewLine)
    throw "Python is missing controller dependencies. Run SETUP-HOST.bat or: $Python -m pip install -r python\requirements.txt"
}

# --- Godot ------------------------------------------------------------------
if (-not $packaged) {
    if (-not $Godot) {
        if (Get-Command godot -ErrorAction SilentlyContinue) {
            $Godot = (Get-Command godot).Source
        } elseif (Get-Command godot4 -ErrorAction SilentlyContinue) {
            $Godot = (Get-Command godot4).Source
        } elseif (Test-Path -LiteralPath (Join-Path $root "godot\godot.exe")) {
            $Godot = Join-Path $root "godot\godot.exe"
        } else {
            throw "Godot not found. Install Godot 4.7.2, put it on PATH, or pass -Godot / set GODOT."
        }
    }
    if (-not (Test-Path -LiteralPath $Godot)) {
        throw "Godot executable not found: $Godot"
    }
}

# --- Model and tokenizer ----------------------------------------------------
# An explicitly named model or tokenizer is authoritative. Falling back to a
# different checkpoint would silently change the provenance recorded in every
# decision, so a bad explicit path is an error.
$explicitModel = [bool]$Model
$explicitTokenizer = [bool]$Tokenizer
$explicitData = [bool]$Data

if ($explicitModel) {
    if (-not (Test-Path -LiteralPath $Model)) {
        throw "ONNX model not found: $Model"
    }
    $Model = (Resolve-Path -LiteralPath $Model).Path
} else {
    $modelCandidates = @()
    if ($packaged) {
        $modelCandidates += (Join-Path $root "runtime\kev07b_fp16_L256.onnx")
    }
    # Transfer-package layout: the extracted project and the model folder are
    # siblings under the same transfer directory.
    $siblingModel = Join-Path (Split-Path -Parent $root) "LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx"
    $modelCandidates += @(
        $siblingModel,
        (Join-Path $root "models\kev07b_fp16_L256.onnx"),
        (Join-Path $env:LOCALAPPDATA "LANESHIFT\models\kev07b_fp16_L256.onnx"),
        "C:\LANESHIFT2D-Kev-model\kev07b_fp16_L256.onnx",
        (Join-Path $env:USERPROFILE "dev\kev-onnx-backup\kev07b_fp16_L256.onnx")
    )
    $Model = Resolve-FirstPath -Candidates $modelCandidates -Label "ONNX model (kev07b_fp16_L256.onnx)"
}
$modelDir = Split-Path -Parent $Model

if ($explicitTokenizer) {
    if (-not (Test-Path -LiteralPath $Tokenizer)) {
        throw "Tokenizer not found: $Tokenizer"
    }
    $Tokenizer = (Resolve-Path -LiteralPath $Tokenizer).Path
} else {
    $tokenizerCandidates = @()
    if ($packaged) {
        $tokenizerCandidates += (Join-Path $root "runtime\tokenizer.json")
    }
    $tokenizerCandidates += @(
        (Join-Path $modelDir "tokenizer.json"),
        (Join-Path $root "models\tokenizer.json"),
        (Join-Path $env:LOCALAPPDATA "LANESHIFT\models\tokenizer.json"),
        "C:\LANESHIFT2D-Kev-model\tokenizer.json"
    )
    $Tokenizer = Resolve-FirstPath -Candidates $tokenizerCandidates -Label "tokenizer.json"
}

if ($explicitData) {
    if (-not (Test-Path -LiteralPath $Data)) {
        throw "External weights not found: $Data"
    }
    $Data = (Resolve-Path -LiteralPath $Data).Path
} else {
    $adjacentData = Join-Path $modelDir "kev07b_fp16.data"
    if (Test-Path -LiteralPath $adjacentData) {
        $Data = (Resolve-Path -LiteralPath $adjacentData).Path
    } elseif ($explicitModel) {
        throw "External weights kev07b_fp16.data not found next to the model at $modelDir. Pass -Data or set LANESHIFT_ONNX_DATA."
    } else {
        $dataCandidates = @()
        if ($packaged) {
            $dataCandidates += (Join-Path $root "runtime\kev07b_fp16.data")
        }
        $dataCandidates += @(
            (Join-Path $root "models\kev07b_fp16.data"),
            "C:\LANESHIFT2D-Kev-model\kev07b_fp16.data"
        )
        $Data = Resolve-FirstPath -Candidates $dataCandidates -Label "external weights (kev07b_fp16.data)"
    }
}

if (-not $ModelId) {
    $ModelId = "jaredpalmer/kev-0.8b@v1.0+onnx:L$Length"
}

# --- Mode defaults ----------------------------------------------------------
if ($Episodes -le 0) {
    if ($mode -eq "eval" -or $mode -eq "live") { $Episodes = 1 } else { $Episodes = 999 }
}
if ($EpisodeDuration -le 0) {
    if ($mode -eq "eval") { $EpisodeDuration = 20 } elseif ($mode -eq "watch") { $EpisodeDuration = 15 } else { $EpisodeDuration = 60 }
}
if ($ControllerTimeout -le 0) {
    if ($mode -eq "eval") { $ControllerTimeout = 30.0 } elseif ($mode -eq "live") { $ControllerTimeout = 20.0 } else { $ControllerTimeout = 10.0 }
}
$seedCount = @($Seeds.Split(",") | Where-Object { $_.Trim() -ne "" }).Count
if ($seedCount -lt 1) { $seedCount = 1 }
# Live mode cycles seeds so a crash does not repeat the identical scenario. The
# experiment runner's per-seed --episodes semantics are unchanged in eval/watch.
if ($mode -eq "live") {
    $totalEpisodes = if ($MaxEpisodes -gt 0) { $MaxEpisodes } else { 0 }
} else {
    $totalEpisodes = $seedCount * $Episodes
    if ($MaxEpisodes -gt 0) {
        $totalEpisodes = [Math]::Min($totalEpisodes, $MaxEpisodes)
    }
}

$controllerLog = Join-Path $logDir "kev-controller.log"
$controllerErr = Join-Path $logDir "kev-controller.err.log"
$controllerMetrics = Join-Path $logDir "kev-controller-metrics.json"
Remove-Item $controllerLog, $controllerErr, $controllerMetrics -Force -ErrorAction SilentlyContinue

$controllerArgs = @(
    "server.py",
    "--policy", "onnx_kev",
    "--port", "$Port",
    "--onnx-model", $Model,
    "--onnx-tokenizer", $Tokenizer,
    "--onnx-data", $Data,
    "--onnx-length", "$Length",
    "--onnx-threads", "$Threads",
    "--onnx-provider", $Provider,
    "--onnx-model-id", $ModelId,
    "--onnx-label-source", $LabelSource,
    "--kev-prompt", $PromptMode,
    "--decision-timeout", "$ControllerTimeout")
# A finite run lets the controller finish its episode count, write metrics,
# and exit cleanly. Interactive runs are stopped when the game window closes.
if ($mode -eq "live") {
    if ($MaxEpisodes -gt 0) {
        $controllerArgs += @("--expected-episodes", "$MaxEpisodes", "--metrics-out", $controllerMetrics)
    }
} elseif ($Episodes -lt 999 -and $totalEpisodes -gt 0 -and $totalEpisodes -lt 999) {
    $controllerArgs += @("--expected-episodes", "$totalEpisodes", "--metrics-out", $controllerMetrics)
}
switch ($mode) {
    "live" {
        $controllerArgs += @("--live-mode", "--max-state-age", "30.0")
    }
    "eval" {
        $controllerArgs += @("--eval-mode", "--max-state-age", "60.0")
    }
    "watch" {
        $controllerArgs += @("--max-state-age", "10.0")
        if (-not $NoSafety) {
            $controllerArgs += "--safety-fallback"
        }
    }
}

$gameArgs = @(
    "--auto-run",
    "--seeds=$Seeds",
    "--episodes=$Episodes",
    "--episode-duration=$EpisodeDuration",
    "--ws-port=$Port",
    "--traffic-density=$TrafficDensity",
    "--sim-speed-scale=1.0",
    "--max-fps=$MaxFps"
)
if ($MaxEpisodes -gt 0) { $gameArgs += "--max-episodes=$MaxEpisodes" }
if ($Windowed) { $gameArgs += "--windowed" }
switch ($mode) {
    "live" {
        $gameArgs += @(
            "--live",
            "--cycle-seeds",
            "--decision-interval-s=$DecisionInterval",
            "--max-decision-age-s=$MaxDecisionAge",
            "--request-timeout-ms=$RequestTimeoutMs",
            "--inference-cooldown-ms=$InferenceCooldownMs",
            "--external-timeout-ms=60000",
            "--max-stale-ticks=6000"
        )
    }
    "eval" {
        $gameArgs += @(
            "--turn-based",
            "--eval-mode",
            "--decision-interval=$DecisionInterval",
            "--turn-timeout-ms=$TurnTimeoutMs",
            "--external-timeout-ms=60000",
            "--max-stale-ticks=6000"
        )
    }
    "watch" {
        $gameArgs += @("--external-timeout-ms=12000", "--max-stale-ticks=6000")
    }
}

$controllerArgString = (($controllerArgs | ForEach-Object { Quote-Arg $_ }) -join " ")
Write-Host "LANESHIFT mode: $mode"
Write-Host "Python:  $Python ($pythonVersion, $pythonMachine)"
Write-Host "ONNX:    onnxruntime $ortVersion [$ortProviders], threads $Threads, provider $Provider"
Write-Host "Model:   $ModelId"
Write-Host "Model:   $Model"
Write-Host "Data:    $Data"
if ($mode -eq "live") {
    if ($totalEpisodes -gt 0) {
        Write-Host "Episodes: cycle $seedCount seeds, max $totalEpisodes"
    } else {
        Write-Host "Episodes: cycle $seedCount seeds until the window closes"
    }
} else {
    Write-Host "Episodes: $totalEpisodes ($Episodes per seed x $seedCount seeds)"
}
Write-Host "Logs:    $controllerLog"

if ($DryRun) {
    Write-Host ""
    Write-Host "Controller command:"
    Write-Host ("  " + $Python + " " + $controllerArgString)
    Write-Host ""
    Write-Host "Game command:"
    if ($packaged) {
        Write-Host ("  " + $gameExe + " -- " + ($gameArgs -join " "))
    } else {
        $prefix = if ($Headless) { "--headless " } else { "" }
        Write-Host ("  " + $Godot + " " + $prefix + "--path `"$root`" -- " + ($gameArgs -join " "))
    }
    exit 0
}

# A fresh ZIP extraction has no .godot cache. Godot 4 needs one import pass to
# register global script classes before main.gd can load. This is the same
# import step run_tests.bat performs.
if (-not $packaged) {
    $classCache = Join-Path $root ".godot\global_script_class_cache.cfg"
    if (-not (Test-Path -LiteralPath $classCache)) {
        Write-Host "Preparing the Godot project (first run)..."
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Godot --headless --path $root --editor --quit 2>&1 | ForEach-Object { Write-Host $_ }
        $importExit = $LASTEXITCODE
        $ErrorActionPreference = $previous
        if ($importExit -ne 0) {
            throw "Godot project import failed with exit code $importExit. Run: $Godot --headless --path `"$root`" --editor --quit"
        }
        if (-not (Test-Path -LiteralPath $classCache)) {
            throw "Godot did not create .godot\global_script_class_cache.cfg. Run the editor import manually."
        }
    }
}

Write-Host "Starting local Kev controller on port $Port..."
$controller = Start-Process -FilePath $Python -ArgumentList $controllerArgString -WorkingDirectory $pythonDir -PassThru -WindowStyle Hidden -RedirectStandardOutput $controllerLog -RedirectStandardError $controllerErr

try {
    $deadline = (Get-Date).AddSeconds(180)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($controller.HasExited) {
            $tail = if (Test-Path $controllerErr) { Get-Content $controllerErr -Tail 40 } else { @() }
            throw ("Controller exited before it was ready. See $controllerErr" + [Environment]::NewLine + ($tail -join [Environment]::NewLine))
        }
        if (Test-Path $controllerLog) {
            $listening = Select-String -Path $controllerLog -SimpleMatch "listening at ws://" -ErrorAction SilentlyContinue
            if ($listening) {
                $ready = $true
                break
            }
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        $tail = if (Test-Path $controllerLog) { Get-Content $controllerLog -Tail 40 } else { @() }
        throw ("Controller did not become ready within 180 s. See $controllerLog" + [Environment]::NewLine + ($tail -join [Environment]::NewLine))
    }
    Write-Host "Controller ready. Launching LANESHIFT."

    if ($packaged) {
        $exeArgs = @()
        if ($Headless) { $exeArgs += "--headless" }
        $exeArgs += "--"
        $exeArgs += $gameArgs
        $gameArgString = (($exeArgs | ForEach-Object { Quote-Arg $_ }) -join " ")
        $gameProcess = Start-Process -FilePath $gameExe -ArgumentList $gameArgString -PassThru -NoNewWindow
        Wait-Process -Id $gameProcess.Id
        $gameExit = $gameProcess.ExitCode
    } else {
        $godotArgs = @()
        if ($Headless) { $godotArgs += "--headless" }
        $godotArgs += @("--path", $root, "--") + $gameArgs
        # Godot can write engine warnings to stderr; Windows PowerShell 5.1
        # treats native stderr as terminating under Stop.
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Godot @godotArgs
        $gameExit = $LASTEXITCODE
        $ErrorActionPreference = $previous
    }
    exit $gameExit
} finally {
    if ($controller -and -not $controller.HasExited) {
        Stop-Process -Id $controller.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Controller log: $controllerLog"
    if ((Test-Path $controllerErr) -and ((Get-Item $controllerErr).Length -gt 0)) {
        Write-Host "Controller stderr: $controllerErr"
    }
}
