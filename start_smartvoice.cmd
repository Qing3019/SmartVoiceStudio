

@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
REM 默认运行当前源码。打包完成后传入 --exe 即可选择 GUI EXE。
if defined SMARTVOICE_PYTHON (
    set "LAUNCHER_PYTHON=%SMARTVOICE_PYTHON%"
) else if exist "D:\Anaconda\python.exe" (
    set "LAUNCHER_PYTHON=D:\Anaconda\python.exe"
) else if exist "D:\Anaconda\envs_dir\SmartVoiceStudio\python.exe" (
    set "LAUNCHER_PYTHON=D:\Anaconda\envs_dir\SmartVoiceStudio\python.exe"
) else (
    set "LAUNCHER_PYTHON=python"
)
"%LAUNCHER_PYTHON%" -u "%~dp0launch_smartvoice.py" %*
if errorlevel 1 (
    echo.
    echo Startup failed. Check runtime_logs\cosyvoice_server.log.
    pause
)
endlocal

