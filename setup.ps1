[CmdletBinding()]
param(
    [string]$RepoRoot
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Exe,
        [string[]]$CommandArgs
    )

    $cmd = "$Exe $($CommandArgs -join ' ')"
    Write-Host "Running: $cmd"
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Exe @CommandArgs 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }
    if ($output) { $output | ForEach-Object { Write-Host $_ } }

    if ($exitCode -ne 0) {
        $text = ($output | Out-String)
        $isPipCall = ($CommandArgs -contains "-m") -and ($CommandArgs -contains "pip")
        $isKnownPipSummaryCrash = $text -match "InvalidVersion: Invalid version: '4\.0\.0-unsupported'"
        if ($isPipCall -and $isKnownPipSummaryCrash) {
            Write-Warning "pip hit known metadata-summary crash (pyodbc version metadata) after install. Continuing setup."
            return
        }
        throw "Command failed (exit $exitCode): $cmd"
    }
}

function Try-Step {
    param(
        [string]$Exe,
        [string[]]$CommandArgs
    )

    $cmd = "$Exe $($CommandArgs -join ' ')"
    Write-Host "Running: $cmd"
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Exe @CommandArgs 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }
    if ($output) { $output | ForEach-Object { Write-Host $_ } }
    $text = ($output | Out-String)
    $isPipCall = ($CommandArgs -contains "-m") -and ($CommandArgs -contains "pip")
    if ($isPipCall) {
        $hasKnownBenignSummaryCrash = $text -match "InvalidVersion: Invalid version: '4\.0\.0-unsupported'"
        if ($hasKnownBenignSummaryCrash) {
            Write-Warning "pip hit known metadata-summary crash (pyodbc version metadata) after install. Treating as success."
            return $true
        }

        $hasHardPipFailure = $text -match "(?im)metadata-generation-failed|subprocess-exited-with-error|No matching distribution found|error:\s+invalid command 'bdist_wheel'|Failed to build|Could not build wheels"
        if ($hasHardPipFailure) {
            return $false
        }
    }

    if ($exitCode -eq 0) {
        return $true
    }

    $isKnownPipSummaryCrash = $text -match "InvalidVersion: Invalid version: '4\.0\.0-unsupported'"
    if ($isPipCall -and $isKnownPipSummaryCrash) {
        Write-Warning "pip hit known metadata-summary crash (pyodbc version metadata) after install. Treating as success."
        return $true
    }
    return $false
}

function Ensure-AbaqusPip {
    $probe = & abaqus python -c "import importlib.util; print('HAS_PIP=' + str(bool(importlib.util.find_spec('pip'))))" 2>$null
    if (($probe | Out-String) -match "HAS_PIP=True") {
        return
    }

    Write-Host "ABAQUS pip not found. Bootstrapping with ensurepip..."
    Invoke-Step -Exe "abaqus" -CommandArgs @("python", "-c", "import ensurepip; ensurepip.bootstrap(default_pip=True); print('ENSUREPIP_OK')")
}

function Install-LocalPackage-Python {
    param([string]$Root)

    # Try editable first, then non-editable.
    if (Try-Step -Exe "python" -CommandArgs @("-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "-e", $Root)) {
        return
    }
    if (Try-Step -Exe "python" -CommandArgs @("-m", "pip", "install", "--disable-pip-version-check", "--upgrade", $Root)) {
        return
    }

    # Hard fallback: path hook.
    $userSite = (& python -c "import site; print(site.getusersitepackages())" 2>$null | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($userSite)) {
        throw "Could not determine standard Python user site-packages for fallback install."
    }
    if (-not (Test-Path $userSite)) {
        New-Item -ItemType Directory -Path $userSite -Force | Out-Null
    }
    $pth = Join-Path $userSite "phd_shared_resources_repo.pth"
    Set-Content -Path $pth -Value $Root
    Write-Host "Created fallback path hook: $pth"
}

function Install-LocalPackage-Abaqus {
    param([string]$Root)

    # Abaqus pip often does not support editable installs. Try robust non-editable first.
    $pipOk = $false
    if (Try-Step -Exe "abaqus" -CommandArgs @("python", "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "--no-build-isolation", $Root)) {
        $pipOk = $true
    }
    elseif (Try-Step -Exe "abaqus" -CommandArgs @("python", "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "-e", $Root)) {
        $pipOk = $true
    }

    # Always ensure path hook for reliability across working directories.
    if (-not $pipOk) {
        Write-Warning "Abaqus pip local package install failed. Falling back to .pth path hook."
    }
    $abqUserSite = (& abaqus python -c "import site; print(site.getusersitepackages())" 2>$null | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($abqUserSite)) {
        throw "Could not determine ABAQUS Python user site-packages for fallback install."
    }
    if (-not (Test-Path $abqUserSite)) {
        New-Item -ItemType Directory -Path $abqUserSite -Force | Out-Null
    }
    $pth = Join-Path $abqUserSite "phd_shared_resources_repo.pth"
    Set-Content -Path $pth -Value $Root
    Write-Host "Created fallback ABAQUS path hook: $pth"
}

function Verify-ResourcesImport {
    param(
        [string]$Exe,
        [string[]]$PrefixArgs
    )

    $script = @'
import os
import tempfile
os.chdir(tempfile.gettempdir())
from resources.lattices import Geometry
print("IMPORT_OK")
'@
    $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString() + ".py")
    try {
        Set-Content -Path $tmpPy -Value $script
        Invoke-Step -Exe $Exe -CommandArgs (@($PrefixArgs + @($tmpPy)))
    }
    finally {
        if (Test-Path $tmpPy) {
            Remove-Item -LiteralPath $tmpPy -Force
        }
    }
}

Push-Location $RepoRoot
try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "python command not found on PATH."
    }
    if (-not (Get-Command abaqus -ErrorAction SilentlyContinue)) {
        throw "abaqus command not found on PATH."
    }

    $requirementsPath = Join-Path $RepoRoot "requirements.txt"
    $requirementsAbaqusPath = Join-Path $RepoRoot "requirements-abaqus.txt"
    if (-not (Test-Path $requirementsPath)) {
        throw "Missing requirements file: $requirementsPath"
    }
    if (-not (Test-Path $requirementsAbaqusPath)) {
        throw "Missing requirements file: $requirementsAbaqusPath"
    }

    $originalNoIndex = $null
    if (-not [string]::IsNullOrWhiteSpace($env:PIP_NO_INDEX)) {
        $originalNoIndex = $env:PIP_NO_INDEX
        Write-Warning "PIP_NO_INDEX is set ($originalNoIndex). Temporarily unsetting it for setup."
        Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue
    }

    Write-Host "=== Standard Python: install requirements.txt ==="
    Invoke-Step -Exe "python" -CommandArgs @("-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "-r", $requirementsPath)

    Write-Host "=== Standard Python: enforce NumPy/Pandas binary compatibility ==="
    Invoke-Step -Exe "python" -CommandArgs @("-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "numpy>=1.26,<2", "numexpr>=2.8.4", "bottleneck>=1.3.6")

    Write-Host "=== Standard Python: install local resources package ==="
    Install-LocalPackage-Python -Root $RepoRoot

    Write-Host "=== Verify standard Python import ==="
    Verify-ResourcesImport -Exe "python" -PrefixArgs @()
    Write-Host "PYTHON setup OK"

    Ensure-AbaqusPip

    Write-Host "=== ABAQUS Python: install requirements-abaqus.txt ==="
    Invoke-Step -Exe "abaqus" -CommandArgs @("python", "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "-r", $requirementsAbaqusPath)

    Write-Host "=== ABAQUS Python: install local resources package ==="
    Install-LocalPackage-Abaqus -Root $RepoRoot

    Write-Host "=== Verify ABAQUS Python import ==="
    Verify-ResourcesImport -Exe "abaqus" -PrefixArgs @("python")
    Write-Host "ABAQUS setup OK"

    Write-Host "Setup completed successfully for Python and ABAQUS Python."
}
finally {
    if ($null -ne $originalNoIndex) {
        $env:PIP_NO_INDEX = $originalNoIndex
    }
    Pop-Location
}
