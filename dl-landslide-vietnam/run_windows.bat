@echo off
setlocal
cd /d "%~dp0"
echo DL Landslide Vietnam - Windows local workflow
echo 1. Install NVIDIA CUDA environment
echo 2. Check installed environment
echo 3. Run synthetic smoke test
echo 4. Open Vietnamese instructions
choice /c 1234 /m "Select"
if errorlevel 4 goto guide
if errorlevel 3 goto smoke
if errorlevel 2 goto check
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_windows.ps1" -Compute cuda
goto end
:check
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m landslide check-env
goto end
:smoke
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m unittest discover -s tests -p test_end_to_end.py -v
goto end
:guide
start "" notepad.exe "docs\translations\README.vi.md"
goto end
:missing
echo Project environment is missing. Run option 1 first.
:end
pause
