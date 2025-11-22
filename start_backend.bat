@echo off
echo Starting Wafer Defect Analysis Backend...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Set environment variables
set HF_API_KEY=your_huggingface_api_key_here
set MODEL_CACHE_DIR=./models_cache
set LOG_LEVEL=INFO

REM Start FastAPI server
python -m uvicorn app.main:app --reload --port 8001

pause

