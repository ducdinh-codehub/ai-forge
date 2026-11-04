# AIForge installer for Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/install.ps1 | iex
#
# What it does, and nothing else (no admin rights, no system files):
#   1. installs uv (https://docs.astral.sh/uv/) if it is missing: it brings its own Python
#   2. installs AIForge with uv, in its own environment (run it again to update)
#   3. adds AIForge to the Start Menu and the Desktop
# Your models in %USERPROFILE%\AIForge are never touched.
#
# Uninstall (keeps your models):
#   $env:AIFORGE_UNINSTALL = 1; irm https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/install.ps1 | iex
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$Package = "aiforge-studio"
$Python = "3.12"
# Where the shortcuts go (tests point these somewhere else)
$StartMenu = if ($env:AIFORGE_APPS_DIR) { $env:AIFORGE_APPS_DIR } else { Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs" }
$Desktop = if ($env:AIFORGE_APPS_DIR) { $env:AIFORGE_APPS_DIR } else { [Environment]::GetFolderPath("Desktop") }

function Step($text) { Write-Host ""; Write-Host $text -ForegroundColor Cyan }

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Remove-AIForge {
    Step "Uninstalling AIForge"
    $uv = Find-Uv
    if ($uv -and ((& $uv tool list 2>$null) -match "^$Package ")) {
        & $uv tool uninstall $Package
    } else {
        Write-Host "AIForge isn't installed with uv: nothing to remove there."
    }
    foreach ($dir in @($StartMenu, $Desktop)) {
        Remove-Item (Join-Path $dir "AIForge.lnk") -ErrorAction SilentlyContinue
    }
    Write-Host "Done. Your models in $env:USERPROFILE\AIForge were kept (delete that folder yourself if you want them gone)."
    Write-Host "uv itself stays installed; remove it with 'uv self uninstall' if you don't use it."
}

function Add-Shortcut($dir, $exe) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path $dir "AIForge.lnk"))
    $wt = Get-Command wt.exe -ErrorAction SilentlyContinue
    if ($wt) {  # Windows Terminal, in a window big enough for the app
        $link.TargetPath = $wt.Source
        $link.Arguments = "--size 130,40 `"$exe`""
    } else {    # the classic console, resized first
        $link.TargetPath = $env:ComSpec
        $link.Arguments = "/c mode con: cols=130 lines=40 & `"$exe`""
    }
    $link.WorkingDirectory = $env:USERPROFILE
    $link.IconLocation = $exe
    $link.Description = "Build, train and use neural networks without code"
    $link.Save()
}

function Install-AIForge {
    Write-Host "This installs AIForge for you only (no admin rights needed)."
    Write-Host "The first time it downloads Python and PyTorch: about 1 GB, a few minutes."

    $uv = Find-Uv
    if (-not $uv) {
        Step "1/3 - Installing uv (it brings its own Python)"
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        $uv = Find-Uv
        if (-not $uv) { throw "uv was installed but can't be found. Open a new PowerShell window and run this again." }
    } else {
        Step "1/3 - uv is already installed ($uv)"
    }

    Step "2/3 - Installing AIForge"
    & $uv tool install --upgrade --python $Python $Package
    if ($LASTEXITCODE -ne 0) { throw "Installing AIForge failed (see the messages above)." }
    $bin = (& $uv tool dir --bin).Trim()
    $exe = Join-Path $bin "aiforge.exe"
    if (-not (Test-Path $exe)) { throw "The install finished but $exe is missing." }
    if (-not (($env:PATH -split ";") -contains $bin)) {
        & $uv tool update-shell 2>$null | Out-Null  # so 'aiforge' also works in new windows
    }

    Step "3/3 - Adding AIForge to the Start Menu and the Desktop"
    Add-Shortcut $StartMenu $exe
    Add-Shortcut $Desktop $exe

    Step "Done: $(& $exe --version) is installed"
    Write-Host "Start it from the AIForge icon, or type 'aiforge' in a new window."
    Write-Host "New here? Press F2 in the app for the guided tour."
}

if ($Uninstall -or $env:AIFORGE_UNINSTALL -eq "1") { Remove-AIForge } else { Install-AIForge }
