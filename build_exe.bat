@echo off
rem Builds dist\Jade_Studio.exe from jade_gui.py + jade.py (same folder).
rem Needs Python 3.10+ installed (python.org, tick "Add to PATH").
rem Takes a few minutes the first time.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

if not exist .build-venv (
    echo Creating build environment...
    %PY% -m venv .build-venv || goto :err
)
call .build-venv\Scripts\activate.bat || goto :err

python -m pip install --upgrade pip >nul
python -m pip install pyserial numpy PySide6 pyqtgraph scipy pyinstaller || goto :err

pyinstaller --noconfirm --onefile --windowed --name Jade_Studio ^
    --exclude-module matplotlib --exclude-module tkinter ^
    jade_gui.py || goto :err

echo.
echo Done: %~dp0dist\Jade_Studio.exe
echo You can copy that single file anywhere, Python is not needed to run it.
pause
exit /b 0

:err
echo.
echo Build failed, see the messages above.
pause
exit /b 1
