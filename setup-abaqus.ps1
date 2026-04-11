[CmdletBinding()]
param(
    [string]$RepoRoot
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$ErrorActionPreference = 'Stop'

function Set-SessionPythonPath {
    param(
        [string]$Root,
        [string]$DepsDir
    )

    $parts = @()
    if ($DepsDir) { $parts += $DepsDir }
    $parts += $Root

    $current = $env:PYTHONPATH
    foreach ($p in $parts) {
        if ([string]::IsNullOrWhiteSpace($current)) {
            $current = $p
        } elseif ($current -notmatch [Regex]::Escape($p)) {
            $current = "$p;$current"
        }
    }

    $env:PYTHONPATH = $current
    Write-Host "Session PYTHONPATH: $env:PYTHONPATH"
}

function Invoke-AbaqusPython {
    param([string]$Code)

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & abaqus python -c $Code 2>&1
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }

    if ($output) { $output | ForEach-Object { Write-Host $_ } }

    $text = ($output | Out-String)
    $hasTraceback = $text -match 'Traceback \(most recent call last\):'
    $hasImportError = $text -match 'ModuleNotFoundError|ImportError'
    $success = ($exit -eq 0) -and -not $hasTraceback -and -not $hasImportError

    [PSCustomObject]@{
        Success = $success
        ExitCode = $exit
        Output = $text
    }
}

function Invoke-AbaqusPip {
    param([string[]]$PipArgs)

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & abaqus python -m pip @PipArgs 2>&1
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }

    if ($output) { $output | ForEach-Object { Write-Host $_ } }

    $text = ($output | Out-String)
    $usageOnly = ($text -match '(?m)^Usage:\s*$') -and ($text -match '(?m)^Commands:\s*$')
    $pepFailure = $text -match 'subprocess-exited-with-error|metadata-generation-failed'
    $success = ($exit -eq 0) -and -not $usageOnly -and -not $pepFailure

    [PSCustomObject]@{
        Success = $success
        ExitCode = $exit
        Output = $text
    }
}

function Ensure-AbaqusPip {
    $probe = Invoke-AbaqusPython -Code "import importlib.util as u; print('HAS_PIP=' + str(bool(u.find_spec('pip'))))"
    if ($probe.Success -and $probe.Output -match 'HAS_PIP=True') {
        return $true
    }

    Write-Host 'pip not found in ABAQUS Python. Trying ensurepip...'
    $ensure = Invoke-AbaqusPython -Code "import ensurepip; ensurepip.bootstrap(default_pip=True); print('ENSUREPIP_OK')"
    if ($ensure.Success -and $ensure.Output -match 'ENSUREPIP_OK') {
        return $true
    }

    return $false
}

function Install-RequirementsFallback {
    param(
        [string]$RequirementsFile,
        [string]$DepsDir
    )

    New-Item -ItemType Directory -Force -Path $DepsDir | Out-Null

    $requirements = Get-Content $RequirementsFile |
        ForEach-Object { $_.Trim() } |
        Where-Object {
            $_ -and
            -not $_.StartsWith('#') -and
            -not $_.StartsWith('-e') -and
            -not $_.StartsWith('--')
        }

    if ($requirements.Count -eq 0) {
        return $true
    }

    & python -m pip install @requirements -t $DepsDir
    return ($LASTEXITCODE -eq 0)
}

function Try-InstallRepoPth {
    param([string]$Root)

    $siteQuery = Invoke-AbaqusPython -Code "import site; p=[x for x in site.getsitepackages() if x.endswith('site-packages')]; print('SITEPKG=' + (p[0] if p else ''))"
    if (-not $siteQuery.Success) {
        return $false
    }

    $m = [regex]::Match($siteQuery.Output, 'SITEPKG=(.+)')
    if (-not $m.Success) {
        return $false
    }

    $siteDir = $m.Groups[1].Value.Trim()
    if ([string]::IsNullOrWhiteSpace($siteDir)) {
        return $false
    }

    try {
        $pthPath = Join-Path $siteDir 'phd_shared_resources_repo.pth'
        Set-Content -Path $pthPath -Value $Root
        Write-Host "Wrote persistent path file: $pthPath"
        return $true
    }
    catch {
        Write-Warning ("Cannot write .pth in ABAQUS site-packages: " + $_.Exception.Message)
        return $false
    }
}

function Verify-ResourcesImport {
    $tmp = [System.IO.Path]::GetTempPath().TrimEnd('\')
    $check = Invoke-AbaqusPython -Code "import os; os.chdir(r'$tmp'); from resources.lattices import Geometry; import numpy, pandas, matplotlib; print('ABAQUS setup OK')"
    return $check.Success
}

Push-Location $RepoRoot
try {
    $requirementsFile = Join-Path $RepoRoot 'requirements-abaqus.txt'
    if (-not (Test-Path $requirementsFile)) {
        throw "Missing $requirementsFile"
    }

    if (-not (Get-Command abaqus -ErrorAction SilentlyContinue)) {
        throw 'Abaqus command not found on PATH.'
    }

    if (-not (Ensure-AbaqusPip)) {
        throw 'ABAQUS Python pip is unavailable and ensurepip failed.'
    }

    Write-Host 'Installing third-party requirements in ABAQUS Python...'
    $depInstall = Invoke-AbaqusPip -PipArgs @('install', '-r', $requirementsFile)

    $depsDir = $null
    if (-not $depInstall.Success) {
        Write-Warning 'ABAQUS pip install failed. Falling back to local .abaqus-pydeps.'
        $depsDir = Join-Path $RepoRoot '.abaqus-pydeps'
        if (-not (Install-RequirementsFallback -RequirementsFile $requirementsFile -DepsDir $depsDir)) {
            throw 'Fallback dependency install failed.'
        }
        Set-SessionPythonPath -Root $RepoRoot -DepsDir $depsDir
    }

    Write-Host 'Trying editable install of local repo package in ABAQUS Python...'
    $editable = Invoke-AbaqusPip -PipArgs @('install', '--no-build-isolation', '-e', $RepoRoot)

    if (-not $editable.Success) {
        Write-Warning 'Editable install failed. Trying persistent .pth registration in ABAQUS site-packages.'
        $pthOk = Try-InstallRepoPth -Root $RepoRoot
        if (-not $pthOk) {
            Write-Warning 'Persistent .pth registration failed; using current-shell PYTHONPATH fallback.'
            Set-SessionPythonPath -Root $RepoRoot -DepsDir $depsDir
        }
    }

    if (-not (Verify-ResourcesImport)) {
        throw 'Final import verification failed. If using PYTHONPATH fallback, run ABAQUS from this same shell.'
    }

    Write-Host 'ABAQUS setup OK.'
}
finally {
    Pop-Location
}
