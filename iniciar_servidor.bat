@echo off
setlocal EnableExtensions
chcp 65001 >nul
title RAG web - Iniciar servidor

rem ============================================================
rem  Inicia el RAG web sin complicaciones:
rem   - Arranca Ollama con el almacen de modelos correcto
rem     (F:\ollama_models si existe) y lo reinicia solo si el
rem     Ollama que esta corriendo no ve los modelos.
rem   - Levanta el servidor y abre el navegador solo.
rem  Cierra esta ventana (o Ctrl+C) para detener.
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

rem --- buscar ollama (ruta completa) ---
set "OLLAMA="
for /f "delims=" %%i in ('where ollama 2^>nul') do if not defined OLLAMA set "OLLAMA=%%i"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA (
   echo [ERROR] No se encontro Ollama. Instalalo desde https://ollama.com
   pause
   exit /b 1
)

rem --- detectar el almacen de modelos (los del disco F) ---
set "OLLAMA_MODELS="
if exist "F:\ollama_models" set "OLLAMA_MODELS=F:\ollama_models"
if not defined OLLAMA_MODELS if exist "%USERPROFILE%\.ollama\models" set "OLLAMA_MODELS=%USERPROFILE%\.ollama\models"

rem --- cuantas veces hemos reiniciado Ollama (evita bucles) ---
set /a REINT=0

:check_ollama
>nul 2>&1 curl -s --max-time 2 http://127.0.0.1:11434/api/tags
if errorlevel 1 goto reiniciar

:check_model
curl -s --max-time 5 http://127.0.0.1:11434/api/tags | findstr /i /c:"%MODELO%" >nul 2>&1
if not errorlevel 1 goto modelo_ok
if %REINT% GEQ 2 goto modelo_duda
echo [INFO] Ollama responde pero no ve %MODELO%: el almacen de modelos esta mal configurado.
echo   Lo reinicio para que use %OLLAMA_MODELS%...
if not defined OLLAMA_MODELS goto modelo_duda
goto reiniciar

:reiniciar
set /a REINT+=1
if %REINT% GEQ 3 goto ollama_duda
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM "ollama.exe" >nul 2>&1
powershell -NoProfile -Command "Start-Process -FilePath '%OLLAMA%' -ArgumentList 'serve'"

rem --- esperar a que responda (hasta ~30 seg) ---
set /a N=0
:wait
>nul 2>&1 curl -s --max-time 1 http://127.0.0.1:11434/api/tags
if not errorlevel 1 goto check_model
set /a N+=1
if %N% GEQ 30 goto ollama_duda
ping -n 2 127.0.0.1 >nul
goto wait

:ollama_duda
echo [AVISO] No se pudo confirmar que Ollama responda en 127.0.0.1:11434.
echo   Abrelo manualmente; el chat no podra responder sin Ollama.
goto arrancar

:modelo_ok
echo [OK] Ollama listo (modelo %MODELO% disponible).
goto arrancar

:modelo_duda
echo [AVISO] No se encontro el modelo %MODELO% en el almacen de Ollama.
echo   El chat recuperara las fuentes pero tal vez no genere respuesta.

:arrancar
cd /d "%BASE%"
set "EXP_MODEL=%MODELO%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

echo.
echo Levantando el servidor RAG web...
echo   (La primera carga importa PyTorch / sentence-transformers y puede tardar entre
echo    30 segundos y 1 minuto. No cierres esta ventana hasta que diga "corriendo".)
echo Cuando termines, cierra esta ventana para detener.
echo.
%PY% rag_web.py

pause
endlocal