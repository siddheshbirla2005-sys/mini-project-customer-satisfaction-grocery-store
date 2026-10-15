@echo off
setlocal
title Customer Satisfaction Grocery Store AI
cd /d "%~dp0"
echo ==============================================
echo   CUSTOMER SATISFACTION GROCERY STORE AI
echo ==============================================
echo.
where python >nul 2>nul
if errorlevel 1 (
 echo ERROR: Python is not installed or not in PATH.
 echo Install Python and enable "Add Python to PATH".
 pause
 exit /b 1
)
echo [1/3] Installing required packages...
python -m pip install -r requirements.txt
if errorlevel 1 (echo ERROR: Package installation failed.&pause&exit /b 1)
echo [2/3] Training the Machine Learning model...
python train_model.py
if errorlevel 1 (echo ERROR: Model training failed.&pause&exit /b 1)
echo [3/3] Starting the web application...
echo Keep this window open while using the application.
python -m streamlit run app.py
pause
endlocal
