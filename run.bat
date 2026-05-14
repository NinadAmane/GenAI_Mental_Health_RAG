@echo off
echo ===================================================
echo     Mental Health RAG Bot Setup ^& Run Script
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ and try again.
    pause
    exit /b
)

echo [INFO] Installing required dependencies (this may take a while)...
pip install -r requirements.txt

echo.
echo [INFO] Dependencies installed! Starting the Streamlit app...
echo [INFO] The first run will download the AI models to your PC. Please be patient.
echo.

python -m streamlit run app.py

pause
