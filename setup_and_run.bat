@echo off
cd /d "%~dp0"
echo Installing packages (first time only, may take several minutes)...
python -m pip install -r requirements.txt
if errorlevel 1 (echo Install failed. Make sure Python 3.10-3.12 is installed and on PATH. & pause & exit /b 1)
echo Starting the app...
python -m streamlit run app.py
pause
