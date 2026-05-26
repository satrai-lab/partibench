@echo off
setlocal enabledelayedexpansion
set BIN_DIR=%USERPROFILE%\bin

REM ───────────── Banner ─────────────
echo #############################################################
echo #                                                           #
echo #   PPPP    AAAAA   N   N  DDDD   OOOO   RRRR   AAAAA       #
echo #   P   P  A     A  NN  N  D   D O    O  R   R  A   A       #
echo #   PPPP   AAAAAAA  N N N  D   D O    O  RRRR   AAAAA       #
echo #   P      A     A  N  NN  D   D O    O  R  R   A   A       #
echo #   P      A     A  N   N  DDDD   OOOO   R   R  A   A       #
echo #                                                           #
echo #############################################################

REM ───────────── Setup PATH ─────────────
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
set PATH=%BIN_DIR%;%PATH%
set JQ_PATH=%BIN_DIR%\jq.exe
set KIND_PATH=%BIN_DIR%\kind.exe
set KUBECTL_PATH=%BIN_DIR%\kubectl.exe

REM ───────────── Check/Install curl ─────────────
where curl >nul 2>&1 || (
  echo [INFO] curl not found. Installing via PowerShell...
  powershell -Command "Invoke-WebRequest https://curl.se/windows/dl-8.7.1_2/curl-8.7.1_2-win64-mingw.zip -OutFile curl.zip"
  powershell -Command "Expand-Archive -Force curl.zip -DestinationPath curl_tmp"
  copy curl_tmp\curl-*\bin\curl.exe "%BIN_DIR%" >nul
  del curl.zip & rmdir /s /q curl_tmp
)

REM ───────────── Check/Install jq ─────────────
if not exist "%JQ_PATH%" (
  echo [INFO] jq not found. Downloading...
  curl --ssl-no-revoke -Lo "%JQ_PATH%" https://github.com/stedolan/jq/releases/download/jq-1.6/jq-win64.exe
)

REM ───────────── Check/Install kind ─────────────
if not exist "%KIND_PATH%" (
  echo [INFO] kind not found. Downloading...
  curl --ssl-no-revoke -Lo "%KIND_PATH%" https://kind.sigs.k8s.io/dl/v0.23.0/kind-windows-amd64
)

REM ───────────── Check/Install kubectl ─────────────
if not exist "%KUBECTL_PATH%" (
  echo [INFO] kubectl not found. Downloading...
  curl --ssl-no-revoke -Lo "%KUBECTL_PATH%" https://dl.k8s.io/release/v1.30.0/bin/windows/amd64/kubectl.exe
)

REM ───────────── Check/Install Docker ─────────────
where docker >nul 2>&1 || (
  echo [ERROR] Docker CLI not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop
  exit /b 1
)

echo [INFO] All required tools are installed.

REM ───────────── Get Docker Info ─────────────
for /f %%A in ('docker info --format "{{.NCPU}}"') do set CPU_TOTAL=%%A
for /f %%B in ('powershell -command "(docker info --format '{{.MemTotal}}') / 1048576"') do set MEM_TOTAL=%%B
set MEM_TOTAL_CLEAN=%MEM_TOTAL:,=.%  REM Fix comma issue

echo [INFO] Docker CPU cores: %CPU_TOTAL%
echo [INFO] Docker memory (MiB): %MEM_TOTAL%
echo [DEBUG] Normalized memory: %MEM_TOTAL_CLEAN%

REM ───────────── Generate kind config ─────────────
echo [INFO] Generating kind-cluster-config.json...

for /f %%A in ('docker info --format "{{.NCPU}}"') do set cpu_total=%%A
for /f %%B in ('powershell -command "(docker info --format '{{.MemTotal}}') / 1048576"') do set mem_total=%%B
set mem_total_clean=%mem_total:,=.%

"%JQ_PATH%" --argjson cpu_total !cpu_total! --argjson mem_total !mem_total_clean! -f generate_config.jq nodes.json > kind-cluster-config.json

if not exist kind-cluster-config.json (
  echo [ERROR] Failed to generate kind-cluster-config.json
  exit /b 1
)

echo [INFO] kind-cluster-config.json generated.

REM ───────────── Launch cluster ─────────────
set CLUSTER_NAME=pandora-testbed
echo [INFO] Creating Kind cluster "%CLUSTER_NAME%"...

"%KIND_PATH%" get clusters | findstr /C:"%CLUSTER_NAME%" >nul && (
  echo [INFO] Existing cluster found. Deleting...
  "%KIND_PATH%" delete cluster --name %CLUSTER_NAME%
)

"%KIND_PATH%" create cluster --name %CLUSTER_NAME% --config kind-cluster-config.json
if errorlevel 1 (
  echo [ERROR] Kind cluster creation failed.
  exit /b 1
)

"%KUBECTL_PATH%" config use-context kind-%CLUSTER_NAME%
"%KUBECTL_PATH%" wait --for=condition=Ready node --all --timeout=90s

REM ───────────── Apply manifests ─────────────
echo [INFO] Applying Kubernetes manifests...
"%KUBECTL_PATH%" apply -f manifests\00-namespaces
"%KUBECTL_PATH%" apply -f manifests\02-secrets-and-configs
"%KUBECTL_PATH%" apply -f manifests\03-core-infrastructure
"%KUBECTL_PATH%" apply -f manifests\04-data-platform
"%KUBECTL_PATH%" apply -f manifests\05-pipeline-components
echo.
echo [INFO]  Pandora Testbed (Windows) setup complete!
pause
