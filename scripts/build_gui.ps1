# 在项目根目录运行：powershell -ExecutionPolicy Bypass -File scripts/build_gui.ps1
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
Write-Host "构建当前源码版本：" -NoNewline
& python -c "from app_version import APP_VERSION; print(APP_VERSION)"
& python -m PyInstaller --noconfirm --clean SmartVoiceStudio.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败" }
$exe = Join-Path $root "dist\SmartVoiceStudio\SmartVoiceStudio.exe"
if (-not (Test-Path $exe)) { throw "未找到 GUI EXE: $exe" }
Write-Host "构建完成：$exe"
Write-Host "双击运行完整软件：start_smartvoice.cmd --exe"
