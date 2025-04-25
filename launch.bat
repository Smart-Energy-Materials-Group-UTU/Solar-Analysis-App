@echo off
REM Check if the virtual environment exists
IF NOT EXIST "thesis\Scripts\activate" (
    echo Virtual environment not found, setting it up now...
    python -m venv "thesis"
    echo Installing dependencies...
    "thesis\Scripts\pip" install -r "requirements.txt"
)

REM Activate the virtual environment
call "thesis\Scripts\activate.bat"

REM Run the Python application
echo Launching the Solar Analysis Application...
python "Project Code\main_algorithm.py"

REM Deactivate the virtual environment after finishing
deactivate

REM Keep the window open
pause
