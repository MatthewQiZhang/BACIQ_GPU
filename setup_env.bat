@echo off
setlocal EnableDelayedExpansion

echo.
echo ================================================================
echo   BACIQ_GPU Environment Setup
echo   Creates conda environment: BACIQ_GPU
echo   GPU backend: PyTorch + Pyro (CUDA 12.9)
echo ================================================================
echo.

REM ----------------------------------------------------------------
REM  Step 0 — Locate conda
REM ----------------------------------------------------------------
set CONDA_EXE=

for %%P in (
    "%USERPROFILE%\miniconda3\Scripts\conda.exe"
    "%USERPROFILE%\anaconda3\Scripts\conda.exe"
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
    "C:\miniconda3\Scripts\conda.exe"
    "C:\anaconda3\Scripts\conda.exe"
) do (
    if exist %%P (
        set CONDA_EXE=%%P
        goto :found_conda
    )
)

echo ERROR: conda not found.
echo Please install Miniconda (https://docs.conda.io/en/latest/miniconda.html)
echo and re-run this script.
pause
exit /b 1

:found_conda
echo [OK] Found conda at: %CONDA_EXE%
echo.

REM ----------------------------------------------------------------
REM  Step 1 — Create conda environment (Python 3.11)
REM ----------------------------------------------------------------
echo [1/7] Creating environment 'BACIQ_GPU' (Python 3.11)...
%CONDA_EXE% create -n BACIQ_GPU python=3.11 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause & exit /b 1
)
echo [OK] Environment created.
echo.

REM ----------------------------------------------------------------
REM  Step 2 — conda-forge packages (numpy/pandas come with MKL)
REM ----------------------------------------------------------------
echo [2/7] Installing conda-forge packages (pandas, click, pytest, pytest-mock)...
%CONDA_EXE% install -n BACIQ_GPU -c conda-forge pandas click pytest pytest-mock -y
if errorlevel 1 (
    echo ERROR: conda-forge install failed.
    pause & exit /b 1
)
echo [OK] conda-forge packages installed.
echo.

REM ----------------------------------------------------------------
REM  Step 3 — PyTorch with CUDA 12.9
REM  NOTE: must be done BEFORE pyro-ppl so pyro links against CUDA torch
REM ----------------------------------------------------------------
echo [3/7] Installing PyTorch (CUDA 12.9)...
echo        This may take a few minutes (large download ~2 GB).
%CONDA_EXE% run -n BACIQ_GPU pip install torch --index-url https://download.pytorch.org/whl/cu129
if errorlevel 1 (
    echo ERROR: PyTorch CUDA install failed.
    pause & exit /b 1
)
echo [OK] PyTorch (CUDA 12.9) installed.
echo.

REM ----------------------------------------------------------------
REM  Step 4 — Pyro (PyTorch-based Bayesian inference library)
REM ----------------------------------------------------------------
echo [4/7] Installing Pyro...
%CONDA_EXE% run -n BACIQ_GPU pip install pyro-ppl
if errorlevel 1 (
    echo ERROR: Pyro install failed.
    pause & exit /b 1
)
echo [OK] Pyro installed.
echo.

REM ----------------------------------------------------------------
REM  Step 5 — BACIQ_GPU from GitHub
REM  NOTE: pip resolves torch from PyPI (CPU-only) as a dependency.
REM        Step 6 will re-pin the CUDA version afterwards.
REM ----------------------------------------------------------------
echo [5/7] Installing BACIQ_GPU from GitHub...
echo        (torch dependency will be re-pinned to CUDA in Step 6)
%CONDA_EXE% run -n BACIQ_GPU pip install git+https://github.com/MatthewQiZhang/BACIQ_GPU
if errorlevel 1 (
    echo ERROR: BACIQ_GPU install failed.
    pause & exit /b 1
)
echo [OK] BACIQ_GPU installed.
echo.

REM ----------------------------------------------------------------
REM  Step 6 — Re-pin PyTorch CUDA
REM  pip install BACIQ pulls torch from PyPI (CPU build).
REM  Force-reinstall the CUDA build without touching other packages.
REM ----------------------------------------------------------------
echo [6/7] Re-pinning PyTorch to CUDA 12.9 build...
%CONDA_EXE% run -n BACIQ_GPU pip install torch ^
    --index-url https://download.pytorch.org/whl/cu129 ^
    --force-reinstall --no-deps
if errorlevel 1 (
    echo ERROR: PyTorch CUDA re-pin failed.
    pause & exit /b 1
)
echo [OK] PyTorch CUDA re-pinned.
echo.

REM ----------------------------------------------------------------
REM  Step 7 — Fix OpenMP conflict
REM  conda-forge numpy uses Intel libiomp5md.dll; PyTorch uses libomp.dll.
REM  KMP_DUPLICATE_LIB_OK=TRUE suppresses the runtime error safely.
REM ----------------------------------------------------------------
echo [7/7] Setting KMP_DUPLICATE_LIB_OK=TRUE for environment...
%CONDA_EXE% env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n BACIQ_GPU
if errorlevel 1 (
    echo WARNING: Could not set env var automatically.
    echo          Add KMP_DUPLICATE_LIB_OK=TRUE to your system environment variables manually.
)
echo [OK] KMP_DUPLICATE_LIB_OK set.
echo.

REM ----------------------------------------------------------------
REM  Verification
REM ----------------------------------------------------------------
echo ================================================================
echo   Verifying installation...
echo ================================================================

set VERIFY_SCRIPT=%TEMP%\baciq_verify.py
(
    echo import torch
    echo import pyro
    echo from baciq import inference_methods
    echo import pandas as pd
    echo print^("------ Package versions ------"^)
    echo print^(f"Python  : {torch.__version__.__class__.__mro__[0].__module__}"^)
    echo import sys; print^(f"Python  : {sys.version.split^(^)[0]}"^)
    echo print^(f"PyTorch : {torch.__version__}"^)
    echo print^(f"CUDA    : {torch.cuda.is_available^(^)}"^)
    echo if torch.cuda.is_available^(^): print^(f"GPU     : {torch.cuda.get_device_name^(0^)}"^)
    echo print^(f"Pyro    : {pyro.__version__}"^)
    echo print^(f"BACIQ   : OK"^)
    echo print^(^)
    echo print^("------ Quick MCMC smoke test (GPU) ------"^)
    echo model = inference_methods.PYMC_Model^(samples=100, chains=1, tuning=100, channel='ch0'^)
    echo proteins, hist = model.mcmc_sample^(
    echo     pd.DataFrame^({'Protein ID': ['a','b'], 'ch0': [100, 900], 'sum': [1000, 1000]}^),
    echo     bin_width=0.1
    echo ^)
    echo print^(f"Proteins : {list^(proteins^)}"^)
    echo print^(f"Samples  : {hist.sum^(axis=1^)}"^)
    echo print^(^)
    echo print^("SETUP COMPLETE — activate with:  conda activate BACIQ_GPU"^)
) > %VERIFY_SCRIPT%

%CONDA_EXE% run -n BACIQ_GPU python %VERIFY_SCRIPT%
if errorlevel 1 (
    echo.
    echo WARNING: Verification script encountered an error.
    echo          The environment may still be usable. Check output above.
)

del %VERIFY_SCRIPT% 2>nul

echo.
pause
