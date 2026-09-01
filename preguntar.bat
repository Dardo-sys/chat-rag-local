@echo off
setlocal EnableExtensions
chcp 65001 >nul
title RAG - Preguntas sobre el proyecto

rem ============================================================
rem  RAG local: preguntas sobre tu carpeta del proyecto.
rem  Doble clic para usar. Modelo por defecto: gemma2:2b.
rem  Otro modelo:  set EXP_MODEL=nombre  antes de ejecutar.
rem ============================================================

set "BASE=%~dp0"
set "MODELO=gemma2:2b"

rem --- buscar python ---
set "PY=python"
where python >nul 2>&1
if errorlevel 1 set "PY=py"
%PY% --version >nul 2>&1
if errorlevel 1 (
   echo [ERROR] No se encontro Python.
   pause
   exit /b 1
)

rem --- comprobar Ollama ---
>nul 2>&1 curl -s http://127.0.0.1:11434/api/tags
if errorlevel 1 (
   echo [AVISO] No se detecto Ollama en 127.0.0.1:11434.
   echo   Abre Ollama antes de preguntar.
   echo.
)

echo ============================================================
echo   Q&A sobre tu proyecto (RAG local)
echo   Modelo: %MODELO%
echo   Escribe una pregunta y presiona Enter.
echo   Linea vacia = salir.
echo ============================================================
echo.

:loop
set "PREGUNTA="
set /p "PREGUNTA=>> "
if not defined PREGUNTA goto fin
set "PREGUNTA=%PREGUNTA:"=%"
if "%PREGUNTA%"=="" goto fin

rem --- inicio de la busqueda (cronometro) ---
set "T0=%time%"
echo.
echo   [buscando respuesta...]

cd /d "%BASE%"
set "EXP_MODEL=%MODELO%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
%PY% rag_query.py "%PREGUNTA%"

rem --- fin: mostrar tiempo transcurrido ---
set "T1=%time%"
for /f "tokens=1-4 delims=:., " %%a in ("%T0%") do set /a "h0=%%a, m0=%%b, s0=%%c"
for /f "tokens=1-4 delims=:., " %%a in ("%T1%") do set /a "h1=%%a, m1=%%b, s1=%%c"
set /a "ta=(h1*3600+m1*60+s1)-(h0*3600+m0*60+s0)"
if %ta% lss 0 set /a "ta+=86400"
if %ta% lss 10 set "ta=0%ta%"
echo.
echo   [respuesta lista en %ta%s]
echo   ------------------------------------------------------------
echo.
goto loop

:fin
echo.
echo Hasta luego.
endlocal
