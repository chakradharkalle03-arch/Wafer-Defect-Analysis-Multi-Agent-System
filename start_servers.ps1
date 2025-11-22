# Start Backend and Frontend Servers
Write-Host "=== Starting Wafer Defect Analysis System ===" -ForegroundColor Cyan
Write-Host ""

# Stop existing processes
Write-Host "Stopping existing processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*node*" -or $_.ProcessName -like "*uvicorn*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Free ports
Write-Host "Freeing ports..." -ForegroundColor Yellow
$ports = @(8001, 3002, 3000)
foreach ($port in $ports) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

# Set environment variables
$env:HF_API_KEY = "your_huggingface_api_key_here"
$env:PYTHONPATH = $PWD

# Start Backend
Write-Host "`n=== Starting Backend Server ===" -ForegroundColor Green
Write-Host "Port: 8001" -ForegroundColor Yellow
Write-Host "URL: http://localhost:8001" -ForegroundColor White
Write-Host ""

$backendScript = @"
import sys
import os
sys.path.insert(0, r'$PWD')
os.environ['HF_API_KEY'] = 'your_huggingface_api_key_here'
from uvicorn import run
from app.main import app
run(app, host='127.0.0.1', port=8001, reload=True)
"@

$backendScript | Out-File -FilePath "start_backend_temp.py" -Encoding utf8

if (Test-Path "venv\Scripts\python.exe") {
    Start-Process -FilePath "venv\Scripts\python.exe" -ArgumentList "start_backend_temp.py" -WindowStyle Normal
} else {
    Start-Process -FilePath "python" -ArgumentList "start_backend_temp.py" -WindowStyle Normal
}

Start-Sleep -Seconds 5

# Start Frontend
Write-Host "`n=== Starting Frontend Server ===" -ForegroundColor Green
Write-Host "Port: 3002" -ForegroundColor Yellow
Write-Host "URL: http://localhost:3002" -ForegroundColor White
Write-Host ""

Push-Location "frontend"
$env:PORT = "3002"
$env:REACT_APP_API_URL = "http://localhost:8001/api/v1"

Start-Process -FilePath "npm" -ArgumentList "start" -WindowStyle Normal
Pop-Location

Start-Sleep -Seconds 8

# Check servers
Write-Host "`n=== Checking Server Status ===" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/api/v1/health" -TimeoutSec 5 -UseBasicParsing
    Write-Host "✓ Backend: RUNNING (Status $($response.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend: Not responding" -ForegroundColor Red
    Write-Host "  Check the backend terminal window for errors" -ForegroundColor Yellow
}

Write-Host "`n=== Servers Started ===" -ForegroundColor Green
Write-Host "Backend: http://localhost:8001" -ForegroundColor Yellow
Write-Host "API Docs: http://localhost:8001/docs" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:3002" -ForegroundColor Yellow
Write-Host ""

# Open browsers
Start-Process "http://localhost:8001/docs"
Start-Sleep -Seconds 1
Start-Process "http://localhost:3002"

Write-Host "Browser windows opened!" -ForegroundColor Green
Write-Host "`nNote: Servers are running in separate windows." -ForegroundColor Cyan
Write-Host "Close those windows to stop the servers.`n" -ForegroundColor Cyan

