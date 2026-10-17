@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo   OpenSourceHiggsfield Director Studio (#OpenSourceHiggsfield)
echo   Powered by ComfyUI, LTX-2.5 22B, FLUX.3, and comfy-sdk v0.5.0
echo ============================================================================

if /I "%~1"=="--with-comfy-proxy" (
    echo [1/2] Starting comfy-api-proxy on http://127.0.0.1:8189 -> ComfyUI http://127.0.0.1:8188 ...
    start "Comfy API Proxy (:8189)" cmd /c "comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189"
)

echo [Studio] Opening http://127.0.0.1:8080 in your default browser...
start "" "http://127.0.0.1:8080"

echo [Studio] Starting FastAPI + comfy-sdk v0.5.0 Server on http://127.0.0.1:8080 ...
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8080
endlocal
