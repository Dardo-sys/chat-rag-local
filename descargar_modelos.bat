@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Descargar modelos de Ollama para placa basica

set "BASE=%~dp0"
set "PY=python"

where ollama >nul 2>&1
if errorlevel 1 (
   echo [ERROR] No se encontro 'ollama' en el PATH.
   echo   Abre la app Ollama al menos una vez para que se instale.
   pause
   exit /b 1
)

echo ============================================================
echo   Descargar modelos recomendados para placa basica
echo ============================================================
echo.
echo   Modelos recomendados para equipos sin GPU dedicada
echo   (4-8 GB de RAM) - ordenados por equilibrio velocidad/tamano:
echo.
echo   [G] gemma2:2b         1.63 GB  - default, equilibrado
echo   [L] qwen2.5-coder:1.5b  0.99 GB - minimo RAM
echo   [C] gemma3:4b         3.34 GB  - mas calidad (necesita mas RAM)
echo.
echo   A) Descargar TODOS los recomendados
echo   Enter) Descargar solo gemma2:2b (default)
echo   B) Salir sin descargar
echo.
set "OPC="
set /p "OPC=Opcion [A/Enter/B]: "

if /i "%OPC%"=="B" (
   echo.
   echo Saliendo sin descargar.
   pause
   endlocal
   exit /b 0
)

if /i "%OPC%"=="A" (
   echo.
   echo Descargando los 3 modelos recomendados...
   echo.
   call :pull gemma2:2b
   call :pull qwen2.5-coder:1.5b
   call :pull gemma3:4b
   echo.
   echo Listo. Descargaste el pack para placa basica.
   echo Podes verlos en el selector de la web (recargar).
   pause
   endlocal
   exit /b 0
)

if /i "%OPC%"=="L" (
   call :pull qwen2.5-coder:1.5b
   pause
   endlocal
   exit /b 0
)

if /i "%OPC%"=="C" (
   call :pull gemma3:4b
   pause
   endlocal
   exit /b 0
)

rem default (Enter o G): gemma2:2b
call :pull gemma2:2b
pause
endlocal
exit /b 0

:pull
echo.
echo === Descargando: %~1 ===
ollama pull "%~1"
echo.
echo Estado: %~1 terminado (revisa si hubo error arriba).
echo -------------------------------------------------------------
exit /b 0