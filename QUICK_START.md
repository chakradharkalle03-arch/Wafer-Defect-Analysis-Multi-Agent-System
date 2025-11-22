# Quick Start Guide - Wafer Defect Analysis Multi-Agent System

## 🎯 Project Overview

### What is This Project?

The **Wafer Defect Analysis Multi-Agent System** is an advanced AI-powered solution for automated semiconductor wafer inspection, defect detection, classification, root cause analysis, and quality control reporting. This system revolutionizes how semiconductor manufacturers identify, analyze, and resolve wafer defects using cutting-edge multi-agent AI architecture.

### Why This Project Exists

Semiconductor manufacturing is one of the most complex and precision-critical industries. Traditional wafer inspection relies heavily on:
- Manual visual inspection (time-consuming, error-prone)
- Rule-based algorithms (limited flexibility)
- Single-purpose machine learning models (narrow scope)
- Human engineers for root cause analysis (slow, expensive)

**This project solves these challenges by:**
1. **Automating the entire inspection workflow** - From image upload to comprehensive report generation
2. **Using advanced AI reasoning** - LLM-powered root cause analysis that understands process dependencies
3. **Multi-agent coordination** - Specialized AI agents working together intelligently
4. **Real-time analysis** - Fast, accurate defect detection and classification
5. **Comprehensive reporting** - Automated QC reports with actionable insights

### What This Project Does

The system processes semiconductor wafer images through a sophisticated multi-agent pipeline:

1. **Image Analysis** → Detects all defects in wafer images using advanced computer vision
2. **Classification** → Categorizes defects into 8 types (CMP, litho, pattern, scratches, etc.)
3. **Root Cause Analysis** → Identifies which manufacturing process step likely caused each defect
4. **Report Generation** → Creates comprehensive PDF reports with visualizations and recommendations

### How This is an Advanced Project

This system represents **next-generation AI technology** (2024-2025) that is not yet mainstream in semiconductor fabs:

#### 🔥 **Cutting-Edge Features:**

1. **LangGraph Multi-Agent Orchestration**
   - Industry-leading framework for agent coordination
   - Intelligent workflow routing and state management
   - Agent-to-agent communication
   - Used by leading AI companies (OpenAI, Anthropic research)

2. **LLM-Based Root Cause Analysis**
   - Uses Mixtral-8x7B (advanced open-source LLM) for intelligent reasoning
   - Understands process dependencies and defect patterns
   - Similar to research initiatives at TSMC, Samsung, IMEC
   - Not yet widely deployed in production fabs

3. **HuggingFace Inference API Integration**
   - Uses state-of-the-art open-source models:
     - **DETR** (Facebook) for object detection
     - **ViT** (Google) for image classification
   - No local model dependencies - cloud-based, always up-to-date
   - Scalable and maintainable architecture

4. **Advanced Report Generation**
   - LLM-powered executive summaries
   - Intelligent insights extraction
   - Context-aware recommendations
   - Professional PDF generation with visualizations

#### 📊 **Technology Comparison:**

| Component | Traditional Fabs | This System | Status |
|-----------|----------------|-------------|--------|
| Defect Detection | Rule-based / CNN | DETR (Transformer) | ✅ Advanced |
| Classification | Rule-based | ML + Rule Hybrid | ✅ Modern |
| Root Cause Analysis | Human Engineers | **LLM Reasoning** | ⚡ **Next-Gen** |
| Multi-Agent System | Single Models | **LangGraph** | 🔥 **Cutting-Edge** |
| Report Generation | Templates | **LLM-Generated** | ✨ **Emerging** |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.12 recommended)
- **Node.js 16+** (for frontend)
- **HuggingFace API Key** (get one from https://huggingface.co/settings/tokens)
- **Windows/Linux/Mac** (tested on Windows)

### Step 1: Backend Setup

**Windows:**
```powershell
# Navigate to project directory
cd Wafer_Defect_Analysis_Multi_Agent_System

# Activate virtual environment (if not already activated)
.\venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start backend server
.\start_backend.bat
```

**Linux/Mac:**
```bash
# Navigate to project directory
cd Wafer_Defect_Analysis_Multi_Agent_System

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
chmod +x start_backend.sh
./start_backend.sh
```

**Backend will start on:** `http://localhost:8001`

**API Documentation:** `http://localhost:8001/docs`

### Step 2: Frontend Setup

**Open a new terminal:**

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start frontend server
.\start_frontend.bat
```

**Or manually:**
```powershell
cd frontend
$env:PORT="3002"
$env:REACT_APP_API_URL="http://localhost:8001/api/v1"
$env:DANGEROUSLY_DISABLE_HOST_CHECK="true"
npm start
```

**Frontend will start on:** `http://localhost:3002`

### Step 3: Verify System Status

1. **Check Backend Health:**
   - Open: `http://localhost:8001/api/v1/health`
   - Should show: `{"status": "healthy", ...}`

2. **Check Frontend:**
   - Open: `http://localhost:3002`
   - Dashboard should show all agents as ready
   - Models should show: ✓ HF DETR, ✓ HF ViT

### Step 4: Run Your First Analysis

1. **Open the Web Interface:**
   - Navigate to: `http://localhost:3002`

2. **Upload a Wafer Image:**
   - Click or drag-and-drop an image from `test_images/` folder
   - Supported formats: JPG, PNG, TIFF
   - Optionally enter Wafer ID and Batch ID

3. **Wait for Analysis:**
   - Analysis typically takes 10-30 seconds
   - Progress is shown in real-time

4. **Review Results:**
   - **Overview Tab**: Summary statistics
   - **Defects Tab**: Detailed defect information
   - **Root Causes Tab**: Process step analysis
   - **Analytics Tab**: Visual charts

5. **Download Report:**
   - Click "Download Full Report (PDF)"
   - Get comprehensive QC report

---

## 📋 System Architecture

```
┌─────────────────────────────────────────┐
│         React Frontend (Port 3002)      │
│  - Dashboard, Upload, Results Display │
└──────────────────┬──────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      FastAPI Backend (Port 8001)        │
│  - REST API, File Upload, Report Gen    │
└──────────────────┬──────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│   LangGraph Supervisor Agent            │
│   (Multi-Agent Orchestrator)            │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  Image   │ │Classify  │ │  Root    │
│  Agent   │ │  Agent   │ │  Cause   │
│          │ │          │ │  Agent   │
│ DETR/ViT │ │ ViT/ML   │ │  LLM     │
└──────────┘ └──────────┘ └──────────┘
       │          │          │
       └──────────┴──────────┘
                    │
                    ▼
            ┌──────────┐
            │  Report  │
            │  Agent   │
            │   LLM    │
            └──────────┘
```

---

## 🔧 Configuration

### Environment Variables

The system uses the following configuration (already set in `start_backend.bat`):

```powershell
HF_API_KEY=your_huggingface_api_key_here
MODEL_CACHE_DIR=./models_cache
LOG_LEVEL=INFO
```

### Port Configuration

- **Backend:** Port 8001 (configurable in `app/core/config.py`)
- **Frontend:** Port 3002 (configurable via `PORT` environment variable)

---

## 📊 Expected Output

After uploading a wafer image, you should see:

1. **Defect Detection Results:**
   - Total number of defects detected
   - Bounding boxes for each defect
   - Confidence scores

2. **Classification Results:**
   - Defect types (CMP, litho, pattern, scratches, etc.)
   - Classification confidence
   - Defect descriptions

3. **Root Cause Analysis:**
   - Process step identification (CMP, Lithography, Etch, etc.)
   - Likely causes
   - Actionable recommendations

4. **Comprehensive Report:**
   - PDF report with all analysis
   - Visualizations (charts, graphs)
   - Executive summary
   - Detailed findings

---

## 🐛 Troubleshooting

### Backend Won't Start

1. **Check Python version:**
   ```powershell
   python --version  # Should be 3.8+
   ```

2. **Check virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Reinstall dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Frontend Won't Start

1. **Check Node.js version:**
   ```powershell
   node --version  # Should be 16+
   ```

2. **Clear cache and reinstall:**
   ```powershell
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

### Port Already in Use

**Kill process on port 8001:**
```powershell
Get-NetTCPConnection -LocalPort 8001 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
```

**Kill process on port 3002:**
```powershell
Get-NetTCPConnection -LocalPort 3002 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
```

### Models Not Loading

- Verify HuggingFace API key is correct
- Check internet connection (models are loaded via API)
- Check backend logs for errors

---

## 📚 Next Steps

1. **Read the User Manual** (`USER_MANUAL.md`) for detailed usage instructions
2. **Review Agent Report** (`AGENT_REPORT.md`) to understand the AI architecture
3. **Test with different images** from the `test_images/` folder
4. **Explore API documentation** at `http://localhost:8001/docs`

---

## 🎓 Key Technologies

- **LangGraph** - Multi-agent orchestration framework
- **HuggingFace Inference API** - Cloud-based AI models
- **FastAPI** - Modern Python web framework
- **React** - Modern frontend framework
- **Mixtral-8x7B** - Advanced open-source LLM
- **DETR** - Transformer-based object detection
- **ViT** - Vision Transformer for image classification

---

## 📄 License

MIT License - Ready for commercial use

---

**Built with ❤️ for semiconductor manufacturing excellence**
