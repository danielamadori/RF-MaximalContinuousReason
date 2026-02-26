@echo off
REM ==========================================
REM DRIFTS - Docker Management
REM ==========================================

set IMAGE_NAME=drifts:latest
set CONTAINER_NAME=drifts-container

if "%1"=="" goto start
if /I "%1"=="start" goto start
if /I "%1"=="stop" goto stop
if /I "%1"=="restart" goto restart
if /I "%1"=="shell" goto shell
if /I "%1"=="logs" goto logs
if /I "%1"=="rebuild" goto rebuild
if /I "%1"=="clean-rebuild" goto clean_rebuild
if /I "%1"=="help" goto help

:help
echo.
echo Usage: run.bat [command]
echo.
echo Commands:
echo   start     Build and start container (default)
echo   stop      Stop container
echo   restart   Restart container
echo   shell     Open bash shell in container
echo   logs      Show container logs
echo   rebuild   Rebuild image (no cache)
echo   clean-rebuild Remove container and rebuild (no cache)
echo   help      Show this help
echo.
exit /b 0

:start
echo.
echo ==========================================
echo   DRIFTS - Starting Container
echo ==========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found!
    echo Install Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [1/4] Checking Docker... OK
echo.

REM Build image
echo [2/4] Building image %IMAGE_NAME%...
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)
echo Build completed!
echo.

REM Remove existing container
echo [3/4] Removing existing container...
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1
echo.

REM Start container
echo [4/4] Starting container %CONTAINER_NAME%...
docker run -d ^
    --name %CONTAINER_NAME% ^
    -p 6379:6379 ^
    -p 8888:8888 ^
    -v "%cd%\logs:/app/logs" ^
    -v "%cd%\workers:/app/workers" ^
    -v "%cd%\results:/app/results" ^
    -v "%cd%\fig:/app/fig" ^
    %IMAGE_NAME%

if errorlevel 1 (
    echo [ERROR] Failed to start container!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   Container started successfully!
echo ==========================================
echo.
echo Container: %CONTAINER_NAME%
echo Redis:     http://localhost:6379
echo Jupyter:   http://localhost:8888
echo.
echo Commands:
echo   run.bat stop      Stop container
echo   run.bat shell     Open shell
echo   run.bat logs      View logs
echo.
pause
exit /b 0

:stop
echo.
echo Stopping container %CONTAINER_NAME%...
docker stop %CONTAINER_NAME%
echo Container stopped.
echo.
exit /b 0

:restart
echo.
echo Restarting container %CONTAINER_NAME%...
docker restart %CONTAINER_NAME%
echo Container restarted.
echo.
exit /b 0

:shell
echo.
echo Opening shell in %CONTAINER_NAME%...
docker exec -it %CONTAINER_NAME% bash
exit /b 0

:logs
docker logs -f %CONTAINER_NAME%
exit /b 0

:rebuild
echo.
echo ==========================================
echo   DRIFTS - Rebuilding Image
echo ==========================================
echo.
docker build --no-cache -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)
echo Build completed!
goto start

:clean_rebuild
echo.
echo ==========================================
echo   DRIFTS - Clean Rebuild
echo ==========================================
echo.
echo Removing container...
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1
goto rebuild

