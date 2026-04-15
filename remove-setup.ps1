[CmdletBinding()]
param(
    [string]$RepoRoot,
    [switch]$OnlyPython,
    [switch]$OnlyAbaqus,
    [switch]$SkipPythonRequirementsUninstall
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$ErrorActionPreference = "Stop"

if ($OnlyPython -and $OnlyAbaqus) {
    throw "Use only one selector: -OnlyPython or -OnlyAbaqus (not both)."
}

$runPython = $true
$runAbaqus = $true
if ($OnlyPython) {
    $runAbaqus = $false
}
elseif ($OnlyAbaqus) {
    $runPython = $false
}

function Invoke-StepBestEffort {
    param(
        [string]$Exe,
        [string[]]$CommandArgs
    )

    $cmd = "$Exe $($CommandArgs -join ' ')"
    Write-Host "Running: $cmd"
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Command returned non-zero (exit $LASTEXITCODE): $cmd"
    }
}

function Get-RequirementPackages {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return @() }

    return @(Get-Content $Path |
        ForEach-Object { $_.Trim() } |
        Where-Object {
            $_ -and
            -not $_.StartsWith('#') -and
            -not $_.StartsWith('--') -and
            -not $_.StartsWith('-e')
        })
}

function Remove-LegacyPth {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Force
        Write-Host "Removed $Path"
    }
}

function Verify-NotImportable {
    param(
        [string]$Exe,
        [string[]]$PrefixArgs,
        [string]$RepoRootPath
    )

    $script = @'
import importlib.util
import os
import sys
import tempfile

repo_root = os.path.abspath(sys.argv[1]).lower()
os.chdir(tempfile.gettempdir())

clean = []
for p in sys.path:
    if not p:
        continue
    try:
        ap = os.path.abspath(p).lower()
    except Exception:
        clean.append(p)
        continue
    if ap.startswith(repo_root):
        continue
    clean.append(p)
sys.path = clean

print("REMOVED_OK" if importlib.util.find_spec("resources") is None else "STILL_IMPORTABLE")
'@

    $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString() + '.py')
    try {
        Set-Content -Path $tmpPy -Value $script
        $args = @($PrefixArgs + @($tmpPy, $RepoRootPath))
        & $Exe @args
    }
    finally {
        if (Test-Path $tmpPy) {
            Remove-Item -LiteralPath $tmpPy -Force
        }
    }
}

Push-Location $RepoRoot
try {
    $resourceNames = @("phd-shared-resources", "phd_shared_resources", "resources")

    if ($runPython -and (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "=== Standard Python: uninstall local resources package ==="
        foreach ($pkg in $resourceNames) {
            Invoke-StepBestEffort -Exe "python" -CommandArgs @("-m", "pip", "uninstall", "-y", $pkg)
        }

        # Remove fallback path-hook install if present.
        try {
            $pyUserSite = (& python -c "import site; print(site.getusersitepackages())" 2>$null | Select-Object -First 1).Trim()
            if (-not [string]::IsNullOrWhiteSpace($pyUserSite)) {
                Remove-LegacyPth -Path (Join-Path $pyUserSite "phd_shared_resources_repo.pth")
            }
        }
        catch {
            Write-Warning "Could not check standard Python user site-packages for legacy .pth cleanup."
        }

        if (-not $SkipPythonRequirementsUninstall) {
            $reqPath = Join-Path $RepoRoot "requirements.txt"
            $reqPkgs = Get-RequirementPackages -Path $reqPath
            if ($reqPkgs.Count -gt 0) {
                Write-Host "=== Standard Python: uninstall requirements.txt packages ==="
                foreach ($pkg in $reqPkgs) {
                    Invoke-StepBestEffort -Exe "python" -CommandArgs @("-m", "pip", "uninstall", "-y", $pkg)
                }
            }
        }
    }
    elseif ($runPython) {
        Write-Warning "python command not found; skipping standard Python uninstall."
    }

    if ($runAbaqus -and (Get-Command abaqus -ErrorAction SilentlyContinue)) {
        Write-Host "=== ABAQUS Python: uninstall local resources package only ==="
        foreach ($pkg in $resourceNames) {
            Invoke-StepBestEffort -Exe "abaqus" -CommandArgs @("python", "-m", "pip", "uninstall", "-y", $pkg)
        }

        if (-not $SkipPythonRequirementsUninstall) {
            $reqAbaqusPath = Join-Path $RepoRoot "requirements-abaqus.txt"
            $reqAbaqusPkgs = Get-RequirementPackages -Path $reqAbaqusPath
            if ($reqAbaqusPkgs.Count -gt 0) {
                Write-Host "=== ABAQUS Python: uninstall requirements-abaqus.txt packages ==="
                foreach ($pkg in $reqAbaqusPkgs) {
                    Invoke-StepBestEffort -Exe "abaqus" -CommandArgs @("python", "-m", "pip", "uninstall", "-y", $pkg)
                }
            }
        }

        # Optional cleanup of old path-hook based installs:
        try {
            $userSite = (& abaqus python -c "import site; print(site.getusersitepackages())" 2>$null | Select-Object -First 1).Trim()
            if (-not [string]::IsNullOrWhiteSpace($userSite)) {
                Remove-LegacyPth -Path (Join-Path $userSite "phd_shared_resources_repo.pth")
            }
        }
        catch {
            Write-Warning "Could not check ABAQUS user site-packages for legacy .pth cleanup."
        }
    }
    elseif ($runAbaqus) {
        Write-Warning "abaqus command not found; skipping ABAQUS uninstall."
    }

    if ($runPython -and (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "=== Verify standard Python ==="
        Verify-NotImportable -Exe "python" -PrefixArgs @() -RepoRootPath $RepoRoot
    }

    if ($runAbaqus -and (Get-Command abaqus -ErrorAction SilentlyContinue)) {
        Write-Host "=== Verify ABAQUS Python ==="
        Verify-NotImportable -Exe "abaqus" -PrefixArgs @("python") -RepoRootPath $RepoRoot
    }

    Write-Host "Removal script completed."
}
finally {
    Pop-Location
}
