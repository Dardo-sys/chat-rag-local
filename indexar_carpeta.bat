@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Indexar carpeta para el Chat RAG

rem ============================================================
rem  Crea el indice RAG de cualquier carpeta de tu PC.
rem  Luego usa chat_web.bat (o preguntar.bat) para hablar con ella.
rem ============================================================

set "BASE=%~dp0"

set "PY=python"
where python >nul 2>&1
if errorlevel 1 set "PY=py"
%PY% --version >nul 2>&1
if errorlevel 1 (
   echo [ERROR] No se encontro Python.
   pause
   exit /b 1
)

echo ============================================================
echo   Indexar carpeta para el Chat RAG
echo   Escribe la ruta COMPLETA de la carpeta y presiona Enter.
echo   Ejemplo:  C:\Users\tuUsuario\Documentos\mi-proyecto
echo.
echo   (Enter vacio = cancelar)
echo ============================================================
echo.

set "RUTA="
set /p "RUTA=> Ruta: "
if "%RUTA%"=="" (
   echo Cancelado.
   pause
   exit /b 0
)
if not exist "%RUTA%" (
   echo [ERROR] No existe la ruta: "%RUTA%"
   pause
   exit /b 1
)

cd /d "%BASE%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
echo.
echo Indexando: %RUTA%
echo Esto puede tardar la primera vez (carga los embeddings)...
echo.
%PY% rag_index.py --folder "%RUTA%"

echo.
echo Listo. Indice creado. Usa chat_web.bat para hablar con esta carpeta.
pause
endlocal
