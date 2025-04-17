@echo off
setlocal enabledelayedexpansion

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Git is not installed or not in PATH
    exit /b 1
)

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if we're in a git repository
git rev-parse --is-inside-work-tree >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Not in a git repository
    exit /b 1
)

REM Check if there are uncommitted changes
git status --porcelain | findstr /r "^[^ ]" >nul
if %ERRORLEVEL% equ 0 (
    echo There are uncommitted changes. Please commit or stash them first.
    exit /b 1
)

REM Get the new version
set /p VERSION=Enter new version (e.g., 0.1.1): 

REM Get the changes
echo Enter release notes (one line only):
set /p CHANGES=

REM Run the Python script
python create_release.py %VERSION% --changes "%CHANGES%"

endlocal 