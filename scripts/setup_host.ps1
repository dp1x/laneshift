param(
    [string]$Python = "",
    [string]$Venv = "",
    [switch]$Recreate,
    [switch]$AllowEmulated,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Venv) {
    $Venv = Join-Path $root ".venv-host"
}
$requirements = Join-Path $root "python\requirements.txt"
$hostCheck = Join-Path $root "python\tools\host_check.py"

function Get-PythonArch {
    param([string]$Exe)
    try {
        # Windows PowerShell 5.1 turns native stderr into a terminating error
        # when ErrorActionPreference is Stop. A Python wrapper can emit a benign
        # warning there, so relax it around the probe and parse stdout only.
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $result = @(& $Exe -c "import platform; print(platform.machine())" 2>$null)
        $code = $LASTEXITCODE
        $ErrorActionPreference = $previous
        if ($code -ne 0) { return "" }
        $match = @($result | Where-Object { $_ -match '^(ARM64|AMD64|x86|X86)$' } | Select-Object -Last 1)
        if ($match.Count -gt 0) { return ([string]$match[0]).Trim() }
        if ($result.Count -gt 0) { return ([string]$result[-1]).Trim() }
    } catch {
    }
    return ""
}

function Find-Arm64Python {
    # The py launcher can select an ARM64 install explicitly. Prefer an
    # already-native interpreter, then the launcher's ARM64 tags.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $installed = & py -0p 2>&1
        foreach ($line in $installed) {
            if ($line -match "ARM64") {
                if ($line -match "([A-Z]:\\.*python\.exe)") {
                    $candidates += $matches[1]
                }
            }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += (Get-Command python).Source
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            $arch = Get-PythonArch -Exe $candidate
            if ($arch -eq "ARM64") { return $candidate }
        }
    }
    # Try the launcher selectors directly. The selected interpreter reports its
    # own machine value, which is the reliable check.
    foreach ($selector in @("-3.12-arm64", "-3.11-arm64", "-3.12", "-3.11")) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) { break }
        try {
            $previous = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $result = @(& py $selector -c "import platform; print(platform.machine())" 2>$null)
            $code = $LASTEXITCODE
            $ErrorActionPreference = $previous
            if ($code -eq 0 -and @($result | Where-Object { $_ -match '^ARM64$' }).Count -gt 0) {
                return "py $selector"
            }
        } catch {
        }
    }
    return ""
}

$osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
Write-Host "OS architecture: $osArch"
Write-Host "Project root:    $root"

if (-not $Python) {
    $Python = Find-Arm64Python
}
if (-not $Python) {
    if (-not $AllowEmulated) {
        throw @"
No native Windows ARM64 Python was found.

Install Python 3.12 or 3.13 ARM64 from https://www.python.org/downloads/windows/
(choose the Windows installer (ARM64), not the x64 installer), then rerun:
    SETUP-HOST.bat

For a deliberate emulated test only, rerun with:
    powershell -File scripts\setup_host.ps1 -AllowEmulated -Python python
"@
    }
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "No Python found. Install native Windows ARM64 Python 3.12."
    }
    $Python = (Get-Command python).Source
    Write-Warning "Using an emulated or non-ARM64 Python because -AllowEmulated was set. Inference will be slow."
}

$arch = Get-PythonArch -Exe $Python
if ($arch -ne "ARM64" -and -not $AllowEmulated) {
    throw "Selected Python reports '$arch', not ARM64. Install native Windows ARM64 Python, pass -Python with the ARM64 python.exe path, or pass -AllowEmulated for a deliberate emulated test."
}
Write-Host "Python:          $Python ($arch)"

if ($DryRun) {
    Write-Host "Would create venv: $Venv"
    Write-Host "Would install:     $requirements"
    exit 0
}

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Write-Host "Removing existing virtual environment at $Venv"
    Remove-Item -LiteralPath $Venv -Recurse -Force
}
if (-not (Test-Path -LiteralPath $Venv)) {
    Write-Host "Creating virtual environment at $Venv"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -m venv $Venv
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw "Could not create the virtual environment." }
}
$venvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment Python not found: $venvPython"
}

Write-Host "Installing controller dependencies..."
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $venvPython -m pip install --upgrade pip
$pipCode = $LASTEXITCODE
if ($pipCode -eq 0) {
    & $venvPython -m pip install -r $requirements
    $pipCode = $LASTEXITCODE
}
$ErrorActionPreference = $previous
if ($pipCode -ne 0) { throw "Dependency installation failed." }

Write-Host ""
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $venvPython $hostCheck
$checkCode = $LASTEXITCODE
$ErrorActionPreference = $previous
if ($checkCode -ne 0) {
    Write-Warning "The environment check reported a problem; review the output above."
}
Write-Host ""
Write-Host "Setup complete. Start the game with PLAY-KEV.bat."
Write-Host "The launcher prefers $venvPython when it exists."
Write-Host "If the model is not found, set LANESHIFT_ONNX_MODEL and LANESHIFT_ONNX_TOKENIZER."
