@echo off
setlocal

cd /d "%~dp0"

set "BASH_EXE=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH_EXE%" (
    echo Git Bash was not found at "%BASH_EXE%".
    echo Install Git for Windows, or run run_training_pipeline.sh from Git Bash/WSL.
    pause
    exit /b 1
)

echo Starting training pipeline in a new Bash window...
start "Training Pipeline" "%BASH_EXE%" -lc "cd '%~dp0' && bash ./run_training_pipeline.sh; status=$?; echo; echo Pipeline exited with code $status; read -r -p 'Press Enter to close...'"

endlocal