@echo off
setlocal EnableExtensions
chcp 65001 >nul
title RAG web - Iniciar servidor

rem ============================================================
rem  Inicia el RAG web sin complicaciones:
rem   - Verifica dependencias de Python (numpy, sentence-transformers)
rem     y las instala automaticamente si faltan.
rem   - Arranca Ollama si no esta corriendo.
rem   - Verifica si el modelo (gemma2:2b) esta descargado, y si falta
rem     lo descarga automaticamente.
rem   - Levanta el servidor RAG web y abre el navegador solo.
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
   echo ============================================================
   echo [ERROR] No se encontro Python instalado.
   echo ============================================================
   echo Por favor instala Python 3.10 o superior desde https://www.python.org
   echo IMPORTANTE: marca la casilla "Add Python to PATH" durante la instalacion.
   echo.
   pause
   exit /b 1
)

rem --- verificar e instalar dependencias de Python si faltan ---
%PY% -c "import numpy, sentence_transformers" >nul 2>&1
if errorlevel 1 (
   echo ============================================================
   echo   [CONFIGURACION INICIAL] Instalando librerias requeridas
   echo ============================================================
   echo Detectamos que faltan componentes de Python.
   echo Instalando numpy y sentence-transformers via pip...
   echo (Esto se hace una sola vez y puede demorar unos minutos).
   echo.
   if exist "%BASE%requirements.txt" (
      %PY% -m pip install -r "%BASE%requirements.txt"
   ) else (
      %PY% -m pip install numpy sentence-transformers
   )
   if errorlevel 1 (
      echo.
      echo [ERROR] Hubo un problema instalando las librerias.
      echo Intenta ejecutar manualmente en la consola:
      echo   pip install numpy sentence-transformers
      echo.
      pause
      exit /b 1
   )
   echo [OK] Librerias instaladas con exito.
   echo.
)

rem --- buscar ollama (ruta completa) ---
set "OLLAMA="
for /f "delims=" %%i in ('where ollama 2^>nul') do if not defined OLLAMA set "OLLAMA=%%i"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA (
   echo ============================================================
   echo [ERROR] No se encontro Ollama instalado.
   echo ============================================================
   echo Descargalo e instalalo gratis desde https://ollama.com
   echo Una vez instalado, vuelve a hacer doble clic en este archivo.
   echo.
   pause
   exit /b 1
)

rem --- detectar almacen de modelos adicional (disco secundario si existe) ---
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

rem Si Ollama esta corriendo pero no ve el modelo, quizas los modelos estan en F:
if defined OLLAMA_MODELS if exist "%OLLAMA_MODELS%" (
   if %REINT% LSS 2 (
      echo [INFO] Ollama responde pero no ve %MODELO%: reconectando con %OLLAMA_MODELS%...
      goto reiniciar
   )
)

rem Si llego aca y no esta el modelo, lo descargamos automaticamente
goto descargar_modelo

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

:descargar_modelo
echo.
echo ============================================================
echo   [CONFIGURACION INICIAL] Descargando modelo de IA (%MODELO%)
echo ============================================================
echo El modelo '%MODELO%' no esta presente en tu instalacion de Ollama.
echo Para que el chat pueda responder localmente, se descargara ahora
echo (pesa ~1.6 GB y se descarga una unica vez).
echo.
echo Descargando %MODELO%...
"%OLLAMA%" pull "%MODELO%"
if errorlevel 1 (
   echo.
   echo [AVISO] Fallo la descarga automatica de %MODELO%.
   echo   Puedes intentar descargarlo luego con: descargar_modelos.bat
   goto modelo_duda
)
echo [OK] Modelo %MODELO% descargado y listo.
goto modelo_ok

:ollama_duda
echo [AVISO] No se pudo confirmar que Ollama responda en 127.0.0.1:11434.
echo   Abre Ollama manualmente; el chat no podra responder sin el servicio activo.
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

rem --- Gestion inteligente de cache de embeddings ---
set "EMB_CACHE=%USERPROFILE%\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
if exist "%EMB_CACHE%" (
   set "HF_HUB_OFFLINE=1"
   set "TRANSFORMERS_OFFLINE=1"
) else (
   echo.
   echo [INFO] Primera ejecucion: se descargara el modelo de embeddings multilingue (~470 MB).
   echo        Las siguientes veces iniciara 100%% offline de forma instantanea.
   echo.
)

echo.
echo Levantando el servidor RAG web...
echo   (La primera carga importa PyTorch / sentence-transformers y puede tardar entre
echo    30 segundos y 1 minuto. No cierres esta ventana hasta que diga "corriendo".)
echo Cuando termines, cierra esta ventana para detener.
echo.
%PY% rag_web.py

pause
endlocal
