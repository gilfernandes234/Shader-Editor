@echo off
title  Shader Editor Dependencies - Installer
echo ======================================
echo   Shader Editor Dependencies Auto Installer
echo ======================================
echo.

:: -------------------------------
:: Check Python
:: -------------------------------
py --version >nul 2>&1
if errorlevel 1 (
    echo Python not found.
    echo Please install Python 3.10 or newer.
    pause
    exit /b
)

echo Detected Python:
py --version
echo.

:: -------------------------------
:: Check Git
:: -------------------------------
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found.
    echo Please install Git before continuing.
    pause
    exit /b
)

echo Git detected.
echo.

:: -------------------------------
:: Ask user
:: -------------------------------
set /p choice=Do you want to download the project and install dependencies? (Y/N): 
if /I "%choice%" NEQ "Y" (
    echo Installation canceled.
    pause
    exit /b
)

:: -------------------------------
:: Clone repository
:: -------------------------------
if not exist "Image-Processing" (
    echo Cloning repository...
    git clone https://github.com/gBloxy/Image-Processing.git
    if errorlevel 1 (
        echo Failed to clone repository.
        pause
        exit /b
    )
) else (
    echo Repository already exists. Skipping clone.
)

cd Image-Processing

:: -------------------------------
:: Upgrade pip
:: -------------------------------
echo.
echo Upgrading pip...
py -m pip install --upgrade pip

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
