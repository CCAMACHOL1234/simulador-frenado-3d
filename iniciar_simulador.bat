@echo off
chcp 65001 >nul
title Simulador de frenado 3D
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv. Ejecute primero instalar.bat
    pause
    exit /b 1
)
echo Iniciando el simulador... Se abrira el navegador en http://localhost:8501
echo Para cerrar el simulador cierre esta ventana o presione Ctrl + C.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
pause
