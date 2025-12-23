@echo off
title  Shader Editor Dependencies - Installer
echo ======================================
echo   Shader Editor Dependencies Auto Installer
echo ======================================
echo.

:: Check if Python exists
py --version >nul 2>&1
if errorlevel 1 (
    echo Python not found.
    echo Please install Python 3.10 or newer before continuing.
    pause
    exit /b
)

:: Show Python version
echo Detected Python version:
py --version
echo.

:: Ask user
set /p choice=Do you want to install the dependencies? (Y/N): 

if /I "%choice%" NEQ "Y" (
    echo Installation canceled by user.
    pause
    exit /b
)

:: -------------------------------
:: Install dependencies
:: -------------------------------
echo.
echo Installing dependencies...
py -m pip install ^
PyQt6==6.7.1 ^
Pillow==10.4.0 ^
numpy>=1.24.0,<2.0.0 ^
pyqtdarktheme>=2.1.0 ^
pygame ^
moderngl ^
PyOpenGL>=3.1.6 ^
PyOpenGL_accelerate>=3.1.6 ^
qdarktheme>=2.1

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    pause
    exit /b
)

echo.
echo ======================================
echo   Installation completed successfully!
echo ======================================
pause

