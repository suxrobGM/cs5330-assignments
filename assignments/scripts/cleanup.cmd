@echo off

:: Navigate to the parent directory
cd ..

:: Remove the .venv directory if it exists
if exist ".venv" (
    echo Removing .venv directory...
    rmdir /s /q .venv
)

:: Remove all __pycache__ directories
echo Removing all __pycache__ directories...
for /r %%i in (.) do (
    if exist "%%i\__pycache__" (
        rmdir /s /q "%%i\__pycache__"
    )
)

echo Cleanup complete.
