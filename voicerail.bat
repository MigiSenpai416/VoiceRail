@echo off
rem VoiceRail launcher for Windows.
rem First run installs everything (uv, Python 3.12, PyTorch, ...), later runs
rem just start the app. Arguments are passed to `voicerail`, e.g.
rem   voicerail.bat                  start the web UI
rem   voicerail.bat convert song.mp3 -m MyVoice
rem Set VOICERAIL_TORCH=cuda, cuda-legacy or cpu to override GPU detection.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv ^(Python package manager^)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

set "VARIANT=%VOICERAIL_TORCH%"
if "%VARIANT%"=="" (
    set "VARIANT=cpu"
    set "DRIVER=0"
    where nvidia-smi >nul 2>nul
    if not errorlevel 1 (
        for /f "tokens=1 delims=." %%d in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set "DRIVER=%%d"
        rem nvidia-smi prints errors such as "No devices were found" to stdout.
        set "NONNUM="
        for /f "delims=0123456789" %%x in ("!DRIVER!") do set "NONNUM=%%x"
        if defined NONNUM set "DRIVER=0"
        if !DRIVER! GEQ 580 (
            set "VARIANT=cuda"
        ) else if !DRIVER! GEQ 525 (
            set "VARIANT=cuda-legacy"
        ) else (
            echo NVIDIA driver is too old for GPU acceleration; update it for much faster processing.
        )
    )
)

echo VoiceRail: using the '%VARIANT%' PyTorch build.
uv sync --frozen --no-dev --extra %VARIANT%
if errorlevel 1 goto :failed
uv run --no-sync voicerail %*
if errorlevel 1 pause
exit /b

:failed
echo.
echo Installation failed. Check your internet connection and the messages above.
pause
exit /b 1
