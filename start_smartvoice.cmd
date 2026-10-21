@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=D:\Anaconda\envs_dir\SmartVoiceStudio\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] SmartVoiceStudio Python does not exist:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)
"%PYTHON_EXE%" -u "%~dp0launch_smartvoice.py" --source %*
if errorlevel 1 (
    echo.
    echo Startup failed. Read the output above and runtime_logs\cosyvoice_server.log.
    pause
)
