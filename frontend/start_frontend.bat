@echo off
echo Starting Wafer Defect Analysis Frontend...
echo.

set PORT=3002
set REACT_APP_API_URL=http://localhost:8001/api/v1
set DANGEROUSLY_DISABLE_HOST_CHECK=true
set WDS_SOCKET_HOST=localhost
set WDS_SOCKET_PORT=3002

npm start

