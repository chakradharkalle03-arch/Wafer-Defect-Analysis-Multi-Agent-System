# Quick server status check script
Write-Host "`n=== Wafer Defect Analysis System - Server Status ===" -ForegroundColor Cyan
Write-Host ""

# Check Backend
Write-Host "Checking Backend (Port 8001)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri http://localhost:8001/api/v1/health -UseBasicParsing -TimeoutSec 5
    $json = $response.Content | ConvertFrom-Json
    Write-Host "✓ BACKEND: RUNNING" -ForegroundColor Green
    Write-Host "  Status: $($json.status)" -ForegroundColor Gray
    Write-Host "  Version: $($json.version)" -ForegroundColor Gray
    Write-Host "  URL: http://localhost:8001" -ForegroundColor Cyan
    Write-Host "  API Docs: http://localhost:8001/docs" -ForegroundColor Cyan
} catch {
    Write-Host "✗ BACKEND: Not running" -ForegroundColor Red
    Write-Host "  Start it with: .\start_backend.bat" -ForegroundColor Yellow
}

Write-Host ""

# Check Frontend
Write-Host "Checking Frontend (Port 3001)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri http://localhost:3001 -UseBasicParsing -TimeoutSec 3
    Write-Host "✓ FRONTEND: RUNNING" -ForegroundColor Green
    Write-Host "  URL: http://localhost:3001" -ForegroundColor Cyan
    Write-Host "  Open this URL in your browser!" -ForegroundColor Green
} catch {
    Write-Host "✗ FRONTEND: Not running" -ForegroundColor Red
    Write-Host "  Start it with: cd frontend; npm start" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== End of Status Check ===" -ForegroundColor Cyan
Write-Host ""

