@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Chat RAG web - tus archivos

rem ============================================================
rem  Interfaz web tipo chat para el RAG.
rem  Muestra los indices creados y deja elegir con que proyecto hablar.
rem  Enter solo = usa el indice por defecto (el de tu carpeta activa).
rem  Doble clic abre el navegador. Cierra la ventana para detener.
rem ============================================================

set "BASE=%~dp0"
set "MODELO=gemma2:2b"

set "PY=python"
where python >nul 2>&1
if errorlevel 1 set "PY=py"
%PY% --version >nul 2>&1
if errorlevel 1 (
   echo [ERROR] No se encontro Python.
   pause
   exit /b 1
)

>nul 2>&1 curl -s http://127.0.0.1:11434/api/tags
if errorlevel 1 (
   echo [AVISO] No se detecto Ollama en 127.0.0.1:11434.
   echo   Abre Ollama antes de preguntar.
   echo.
)

echo ============================================================
echo   Chat RAG - elige con que proyecto hablar
echo ============================================================
echo.
echo   Indices disponibles:
%PY% -c "import rag_index as R; [print('   -',s) for s in R.list_indices()] if R.list_indices() else print('   (ninguno aun - usa indexar_carpeta.bat)')"
echo.
set "IDX="
set /p "IDX=> Nombre del indice (Enter = por defecto): "
if not "%IDX%"=="" (
   set "RAG_INDEX=%IDX%"
)

cd /d "%BASE%"
set "EXP_MODEL=%MODELO%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
echo.
echo Levantando el chat web (la primera vez carga los embeddings, puede tardar)...
%PY% rag_web.py
pause
endlocal
