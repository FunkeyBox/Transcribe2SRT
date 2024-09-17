@echo off
:: Enable colored output
@echo -------------------------------------------------------------------
@echo SETUP SCRIPT FOR WHISPERX
@echo -------------------------------------------------------------------

:: Check if Conda is installed
echo.
echo Checking if Conda is installed...
where conda >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Conda is not installed. Please install Miniconda or Anaconda.
    exit /b
)

:: Check if the 'whisperx' environment exists
set whisperxExists=false
for /f "tokens=1" %%i in ('conda env list ^| findstr "whisperx"') do (
    set whisperxExists=true
)

IF "%whisperxExists%"=="true" (
    echo [INFO] Conda environment 'whisperx' already exists.
    echo Updating 'whisperx' environment...

    call conda activate whisperx

    echo [INFO] Installing PyTorch with CUDA 11.8...
    call conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

    echo [INFO] Installing/Upgrading whisperx from GitHub...
    pip install git+https://github.com/m-bain/whisperx.git --upgrade

    echo [INFO] Installing dependencies...
    pip install -r requirements.txt

) ELSE (
    echo [INFO] Creating new 'whisperx' environment with Python 3.10...
    call conda create --name whisperx python=3.10 -y

    echo Activating 'whisperx' environment...
    call conda activate whisperx

    echo [INFO] Installing PyTorch with CUDA 11.8...
    call conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

    echo [INFO] Installing whisperx from GitHub...
    pip install git+https://github.com/m-bain/whisperx.git

    echo [INFO] Upgrading whisperx to the latest version...
    pip install git+https://github.com/m-bain/whisperx.git --upgrade

    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo [SUCCESS] Setup is complete!
echo.
@echo -------------------------------------------------------------------
@echo SETUP IS COMPLETE
@echo -------------------------------------------------------------------

pause