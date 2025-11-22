#!/bin/bash

echo "Starting Wafer Defect Analysis Backend..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export HF_API_KEY=your_huggingface_api_key_here
export MODEL_CACHE_DIR=./models_cache
export LOG_LEVEL=INFO

# Start FastAPI server
python -m uvicorn app.main:app --reload --port 8000

