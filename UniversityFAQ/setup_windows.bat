@echo off
cd /d "%~dp0"
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>nul
if errorlevel 1 (
    echo Unsupported Python. Install Python 3.11, 3.12 or 3.13, then rerun.
    pause
    exit /b 1
)
python -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Setup complete. Run start_windows.bat
pause
exit /b 0
:fail
echo Setup failed. Check Python and network.
pause
exit /b 1
