@echo off
chcp 65001 >nul
title Instalacion - Simulador de frenado 3D
cd /d "%~dp0"
echo ============================================================
echo   INSTALACION DEL SIMULADOR DE FRENADO EN PENDIENTE (3D)
echo ============================================================
echo.
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] No se encontro Python.
        echo Instale Python 3.10 o superior desde https://www.python.org/downloads/
        echo y MARQUE la casilla "Add python.exe to PATH" durante la instalacion.
        pause
        exit /b 1
    )
    set "PY=python"
)
echo Python encontrado:
%PY% --version
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
) else (
    echo [1/3] El entorno virtual .venv ya existe.
)
echo [2/3] Actualizando pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo [3/3] Instalando dependencias (puede tardar unos minutos) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt pytest
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias. Revise su conexion a internet.
    pause
    exit /b 1
)
echo.
echo Verificando el modelo matematico con las pruebas automaticas ...
".venv\Scripts\python.exe" -m pytest pruebas -q
echo.
echo ============================================================
echo   INSTALACION TERMINADA. Ejecute iniciar_simulador.bat
echo ============================================================
pause
