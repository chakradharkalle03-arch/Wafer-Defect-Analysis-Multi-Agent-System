# Wafer Defect Analysis Multi-Agent System

A sophisticated AI-powered system for automated wafer inspection, defect classification, root cause analysis, and quality control reporting.

## 🎯 Features

- **Image Agent**: Advanced defect detection using HuggingFace DETR and ViT (via Inference API)
- **Classification Agent**: Intelligent defect type classification (CMP defects, litho hotspots, pattern bridging)
- **Root Cause Agent**: Process step inference and root cause analysis
- **Report Agent**: Automated QC report generation with visualizations

## 🏗️ Architecture

```
┌─────────────────┐
│   React UI      │
│  (Frontend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Multi-Agent Orchestrator      │
│   (LangGraph)                   │
└────────┬────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Image  │ │Classify│ │Root    │ │Report  │
│ Agent  │ │ Agent  │ │Cause   │ │ Agent  │
│        │ │        │ │ Agent  │ │        │
└────────┘ └────────┘ └────────┘ └────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- HuggingFace API key (already configured in `.env`)

### Backend Setup

**Windows:**
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend (API key already in .env)
start_backend.bat
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend
chmod +x start_backend.sh
./start_backend.sh
```

The backend API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

The frontend will be available at `http://localhost:3000`

### Quick Test

1. Open `http://localhost:3000` in your browser
2. Check the System Dashboard - all agents should show as ready
3. Upload a wafer image (JPG, PNG, or TIFF)
4. Wait for analysis (30-60 seconds on first run due to model download)
5. Review results and download the PDF report

### Demo Script

Run the demo script with a test image:
```bash
python demo.py path/to/wafer_image.jpg
```

## 📁 Project Structure

```
.
├── app/
│   ├── agents/
│   │   ├── image_agent.py
│   │   ├── classification_agent.py
│   │   ├── root_cause_agent.py
│   │   └── report_agent.py
│   ├── core/
│   │   ├── config.py
│   │   └── orchestrator.py
│   ├── models/
│   │   └── schemas.py
│   ├── api/
│   │   └── routes.py
│   └── main.py
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── data/
│   ├── raw/
│   └── processed/
├── reports/
└── requirements.txt
```

## 🔧 Configuration

Set your HuggingFace API key in `.env`:
```
HF_API_KEY=your_api_key_here
```

## 📊 Usage

1. Upload wafer images (SEM/optical microscope)
2. System automatically detects defects
3. Classifies defect types
4. Analyzes root causes
5. Generates comprehensive QC reports

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, Node.js
- **AI Models**: HuggingFace Inference API (DETR, ViT, Mixtral-8x7B)
- **Multi-Agent**: LangGraph
- **Vision**: OpenCV, PIL

## 📝 License

MIT License

