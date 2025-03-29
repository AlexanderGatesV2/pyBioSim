@echo off
setlocal

echo Do you want to create a virtual environment? (y/n)
set /p create_venv=

if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    
    echo Activating virtual environment...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else (
        echo Failed to activate virtual environment. Please activate it manually.
        exit /b 1
    )
    
    echo Virtual environment created and activated.
)

echo Installing dependencies...
pip install -r requirements.txt

echo Installing pyBioSim in development mode...
pip install -e .

echo Installation complete!
echo Run 'python main.py' to start the simulation.

pause
