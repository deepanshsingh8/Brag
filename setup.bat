@echo off
echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Creating necessary directories...
mkdir data\processed\text 2>nul
mkdir data\processed\images 2>nul

echo Checking environment variables...
python -c "import os; print('PINECONE_API_KEY:', 'Found' if os.getenv('PINECONE_API_KEY') else 'Missing'); print('ROBOFLOW_API_KEY:', 'Found' if os.getenv('ROBOFLOW_API_KEY') else 'Missing')"

echo Setup complete! You can now run the application with:
echo python src/run_ui.py

pause 