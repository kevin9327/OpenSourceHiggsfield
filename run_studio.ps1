param(
    [switch]$WithComfyProxy = $false,
    [int]$Port = 8080,
    [string]$ComfyUrl = "http://127.0.0.1:8188",
    [int]$ProxyPort = 8189
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "============================================================================" -ForegroundColor Yellow
Write-Host "  OpenSourceHiggsfield Director Studio (#OpenSourceHiggsfield)" -ForegroundColor Yellow
Write-Host "  Powered by ComfyUI, LTX-2.5 22B, FLUX.3, and comfy-sdk v0.5.0" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Yellow

if ($WithComfyProxy -or ($args -contains "--with-comfy-proxy")) {
    Write-Host "[Proxy] Launching comfy-api-proxy on http://127.0.0.1:$ProxyPort -> $ComfyUrl ..." -ForegroundColor Green
    Start-Process -FilePath "comfy-api-proxy" -ArgumentList "--comfyui", $ComfyUrl, "--port", "$ProxyPort" -WindowStyle Minimized
    $env:LOCAL_COMFY_PROXY_URL = "http://127.0.0.1:$ProxyPort"
}

$StudioUrl = "http://127.0.0.1:$Port"
Write-Host "[Studio] Opening $StudioUrl in browser..." -ForegroundColor Green
Start-Job -ScriptBlock {
    param($url)
    Start-Sleep -Seconds 2
    Start-Process $url
} -ArgumentList $StudioUrl | Out-Null

Write-Host "[Studio] Starting FastAPI + comfy-sdk v0.5.0 backend on $StudioUrl ..." -ForegroundColor Green
python -m uvicorn backend.server:app --host 127.0.0.1 --port $Port
