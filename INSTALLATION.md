# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn
- Git

## Backend Setup

### 1. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file in the root directory (already created with your API key):
```
HF_API_KEY=your_huggingface_api_key_here
MODEL_CACHE_DIR=./models_cache
LOG_LEVEL=INFO
BACKEND_URL=http://localhost:8000
```

### 4. Create Required Directories

```bash
mkdir -p data/raw data/processed reports models_cache
```

### 5. Start Backend Server

**Windows:**
```bash
start_backend.bat
```

**Linux/Mac:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

Or manually:
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Start Development Server

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## First Run Notes

1. **Model Download**: On first run, the system will download models from HuggingFace. This may take several minutes depending on your internet connection.

2. **CUDA**: If you have a CUDA-compatible GPU, set `CUDA_AVAILABLE=true` in `.env` for faster processing.

3. **Memory**: Ensure you have at least 8GB RAM available for model loading.

## Testing the System

1. Open the frontend at `http://localhost:3000`
2. Check the System Dashboard to verify all agents are ready
3. Upload a wafer image (JPG, PNG, or TIFF)
4. Wait for analysis to complete
5. Review results and download the PDF report

## Troubleshooting

### Models Not Loading
- Check your internet connection
- Verify HuggingFace API key is correct
- Check `models_cache` directory has write permissions

### Port Already in Use
- Change port in `start_backend.bat`/`start_backend.sh`
- Or kill the process using the port

### Frontend Can't Connect to Backend
- Ensure backend is running on port 8000
- Check CORS settings in `app/main.py`
- Verify `REACT_APP_API_URL` in frontend if using custom URL

## Production Deployment

For production deployment:

1. Set `LOG_LEVEL=WARNING` in `.env`
2. Use a production WSGI server like Gunicorn
3. Configure proper CORS origins
4. Set up reverse proxy (nginx)
5. Use environment variables for sensitive data
6. Enable HTTPS

