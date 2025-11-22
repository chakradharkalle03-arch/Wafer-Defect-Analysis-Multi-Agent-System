# User Manual - Wafer Defect Analysis Multi-Agent System

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Starting the System](#starting-the-system)
5. [Using the Web Interface](#using-the-web-interface)
6. [Understanding Results](#understanding-results)
7. [API Usage](#api-usage)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Introduction

This user manual provides step-by-step instructions to use the Wafer Defect Analysis Multi-Agent System. By following this guide, you will be able to:

- Set up the system from scratch
- Upload and analyze wafer images
- Interpret analysis results
- Generate comprehensive reports
- Reproduce the exact output shown in the system

---

## System Requirements

### Software Requirements

- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** 3.8 or higher (3.12 recommended)
- **Node.js:** 16.0 or higher
- **npm:** 8.0 or higher (comes with Node.js)
- **Internet Connection:** Required for HuggingFace API access

### Hardware Requirements

- **RAM:** Minimum 4GB, Recommended 8GB+
- **Storage:** Minimum 2GB free space
- **CPU:** Any modern processor (multi-core recommended)

---

## Installation

### Step 1: Verify Prerequisites

**Check Python:**
```powershell
python --version
# Should output: Python 3.8.x or higher
```

**Check Node.js:**
```powershell
node --version
# Should output: v16.x.x or higher
```

**Check npm:**
```powershell
npm --version
# Should output: 8.x.x or higher
```

### Step 2: Clone/Download Project

If you have the project files, navigate to the project directory:
```powershell
cd Wafer_Defect_Analysis_Multi_Agent_System
```

### Step 3: Set Up Backend

**Create Virtual Environment (if not exists):**
```powershell
python -m venv venv
```

**Activate Virtual Environment:**

*Windows:*
```powershell
.\venv\Scripts\Activate.ps1
```

*Linux/Mac:*
```bash
source venv/bin/activate
```

**Install Python Dependencies:**
```powershell
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed fastapi-0.104.1 langgraph-0.0.20 ...
```

### Step 4: Set Up Frontend

**Navigate to Frontend Directory:**
```powershell
cd frontend
```

**Install Node Dependencies:**
```powershell
npm install
```

**Expected Output:**
```
added 1234 packages in 45s
```

**Return to Project Root:**
```powershell
cd ..
```

---

## Starting the System

### Method 1: Using Batch Scripts (Recommended)

**Start Backend:**
```powershell
.\start_backend.bat
```

**Expected Output:**
```
INFO: Uvicorn running on http://127.0.0.1:8001
INFO: Application startup complete.
```

**Start Frontend (New Terminal):**
```powershell
cd frontend
.\start_frontend.bat
```

**Expected Output:**
```
Compiled successfully!
You can now view wafer-defect-analysis-frontend in the browser.
  http://localhost:3002
```

### Method 2: Manual Start

**Backend:**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set environment variables
$env:HF_API_KEY="your_huggingface_api_key_here"
$env:MODEL_CACHE_DIR=".\models_cache"
$env:LOG_LEVEL="INFO"

# Start server
python -m uvicorn app.main:app --reload --port 8001
```

**Frontend:**
```powershell
cd frontend
$env:PORT="3002"
$env:REACT_APP_API_URL="http://localhost:8001/api/v1"
$env:DANGEROUSLY_DISABLE_HOST_CHECK="true"
npm start
```

### Step 3: Verify System is Running

**Check Backend:**
1. Open browser: `http://localhost:8001/api/v1/health`
2. Should see: `{"status":"healthy","version":"1.0.0",...}`

**Check Frontend:**
1. Open browser: `http://localhost:3002`
2. Should see the dashboard with system status

---

## Using the Web Interface

### Step 1: Access the Dashboard

1. Open your web browser
2. Navigate to: `http://localhost:3002`
3. You should see:
   - **System Dashboard** at the top
   - **Upload Wafer Image** section in the middle
   - Empty **Analysis Results** section at the bottom

### Step 2: Check System Status

**On the Dashboard, verify:**
- ✓ System Status: **HEALTHY**
- ✓ All agents show as ready:
  - Supervisor Agent: ✓
  - Image Agent: ✓
  - Classification Agent: ✓
  - Root Cause Agent: ✓
  - Report Agent: ✓
- ✓ Models Status:
  - HF DETR: ✓
  - HF ViT: ✓
- ✓ Message: "Using HuggingFace Inference API (Open Source Models)"

### Step 3: Upload a Wafer Image

**Option A: Using Test Images (Recommended for First Time)**

1. Navigate to the `test_images/` folder in the project
2. Select an image (e.g., `wafer_CMP_defects_02.png`)
3. Drag and drop it into the upload area, OR
4. Click the upload area and select the file

**Option B: Using Your Own Image**

1. Ensure image is in format: **JPG, PNG, or TIFF**
2. Drag and drop into upload area
3. Image will be uploaded automatically

**Optional Fields:**
- **Wafer ID:** Enter identifier (e.g., "WAFER001")
- **Batch ID:** Enter batch identifier (e.g., "BATCH001")

### Step 4: Wait for Analysis

**What Happens:**
1. Image is uploaded to backend
2. Analysis workflow starts automatically
3. Progress indicator shows "Analyzing..."
4. Analysis typically takes **10-30 seconds**

**During Analysis:**
- Image Agent detects defects
- Classification Agent categorizes defects
- Root Cause Agent analyzes causes
- Report Agent generates report

### Step 5: Review Results

After analysis completes, the **Analysis Results** section will appear with 4 tabs:

#### **Overview Tab** (Default)

**You will see:**
- **Total Defects:** Number (e.g., "9")
- **Severity Score:** Percentage with badge (e.g., "50.4%" - HIGH)
- **Defect Types:** Count of different types (e.g., "2")
- **Analysis ID:** Unique identifier

**Defect Type Summary:**
- List of defect types with counts
- Example: "Pattern Defects: 8", "Scratches: 1"

**Download Button:**
- Purple button: "📄 Download Full Report (PDF)"
- Click to download comprehensive report

#### **Defects Tab**

**For each defect, you will see:**
- **Defect ID:** Unique identifier
- **Defect Type:** Category (e.g., "Pattern Defects")
- **Confidence:** Percentage (e.g., "85.2%")
- **Location:** Bounding box coordinates
- **Area:** Size in pixels²
- **Description:** Detailed description (if available)

#### **Root Causes Tab**

**For each root cause, you will see:**
- **Defect ID:** Associated defect
- **Process Step:** Manufacturing step (e.g., "CMP", "Lithography")
- **Likely Cause:** Explanation
- **Confidence:** Percentage
- **Recommendations:** Actionable steps (if available)

#### **Analytics Tab**

**Visual Charts:**
1. **Defect Type Distribution:** Pie chart showing defect type percentages
2. **Defects by Process Step:** Bar chart showing process step breakdown
3. **Classification Confidence:** Bar chart showing confidence levels

### Step 6: Download Report

1. Click **"📄 Download Full Report (PDF)"** button
2. PDF will download to your default download folder
3. Open PDF to view:
   - Executive summary
   - Defect analysis
   - Root cause analysis
   - Visualizations
   - Recommendations

---

## Understanding Results

### Severity Score Interpretation

| Score Range | Level | Meaning | Action |
|------------|-------|---------|--------|
| 0.0 - 0.3 | LOW | Minor defects, acceptable quality | Monitor |
| 0.3 - 0.5 | MODERATE | Some defects, review recommended | Review process |
| 0.5 - 0.8 | HIGH | Significant defects, action required | Investigate immediately |
| 0.8 - 1.0 | CRITICAL | Major defects, immediate action needed | Stop production |

### Defect Types

1. **CMP Defects** - Chemical Mechanical Polishing issues
2. **Litho Hotspots** - Lithography process problems
3. **Pattern Bridging** - Unintended connections
4. **Scratches** - Mechanical damage
5. **Particles** - Contamination
6. **Pattern Defects** - Pattern-related issues
7. **Etch Defects** - Etching process problems
8. **Deposition Defects** - Deposition process issues

### Process Steps

- **CMP** - Chemical Mechanical Polishing
- **Lithography** - Pattern transfer
- **Etch** - Material removal
- **Deposition** - Material addition
- **Implant** - Ion implantation
- **Cleaning** - Wafer cleaning

---

## API Usage

### Using cURL

**Analyze an Image:**
```bash
curl -X POST "http://localhost:8001/api/v1/analyze" \
  -F "file=@test_images/wafer_CMP_defects_02.png" \
  -F "wafer_id=WAFER001" \
  -F "batch_id=BATCH001"
```

**Expected Response:**
```json
{
  "analysis_id": "bd1fd190-3933-4a85-9080-ae00245122a",
  "wafer_id": "WAFER001",
  "total_defects": 9,
  "severity_score": 0.504,
  "defect_summary": {
    "pattern_defects": 8,
    "scratches": 1
  },
  ...
}
```

### Using Python

**Create a script `test_api.py`:**
```python
import requests

url = "http://localhost:8001/api/v1/analyze"
files = {"file": open("test_images/wafer_CMP_defects_02.png", "rb")}
data = {"wafer_id": "WAFER001", "batch_id": "BATCH001"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Total defects: {result['total_defects']}")
print(f"Severity score: {result['severity_score']}")
print(f"Defect summary: {result['defect_summary']}")
```

**Run:**
```powershell
python test_api.py
```

**Expected Output:**
```
Total defects: 9
Severity score: 0.504
Defect summary: {'pattern_defects': 8, 'scratches': 1}
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('wafer_id', 'WAFER001');

fetch('http://localhost:8001/api/v1/analyze', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Total defects:', data.total_defects);
  console.log('Severity score:', data.severity_score);
});
```

---

## Troubleshooting

### Problem: Backend Won't Start

**Symptoms:**
- Error: "Port 8001 already in use"
- Error: "Module not found"

**Solution 1: Port Already in Use**
```powershell
# Find and kill process
Get-NetTCPConnection -LocalPort 8001 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
```

**Solution 2: Module Not Found**
```powershell
# Reinstall dependencies
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Problem: Frontend Won't Start

**Symptoms:**
- Error: "Port 3002 already in use"
- Error: "Invalid options object"

**Solution 1: Port Already in Use**
```powershell
# Find and kill process
Get-NetTCPConnection -LocalPort 3002 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
```

**Solution 2: Invalid Options Object**
```powershell
# Use the start_frontend.bat script which sets proper environment variables
cd frontend
.\start_frontend.bat
```

### Problem: Analysis Fails

**Symptoms:**
- Error: "Analysis failed"
- No results displayed

**Solution:**
1. Check backend logs for error messages
2. Verify image format (JPG, PNG, TIFF)
3. Check HuggingFace API key is correct
4. Verify internet connection

### Problem: Models Not Loading

**Symptoms:**
- Dashboard shows models as not loaded
- Analysis returns errors

**Solution:**
1. Verify HuggingFace API key in `start_backend.bat`
2. Check internet connection
3. Check backend logs for API errors
4. Restart backend server

---

## Best Practices

### Image Quality

1. **Use High Resolution:** Higher resolution = better detection
2. **Good Contrast:** Ensure defects are visible
3. **Avoid Compression:** Use lossless formats when possible
4. **Proper Format:** JPG, PNG, or TIFF only

### Batch Processing

1. **Use Consistent IDs:** Use same Batch ID format
2. **Track Wafer IDs:** Maintain consistent naming
3. **Organize Results:** Download and save reports
4. **Monitor Trends:** Track defect patterns over time

### System Maintenance

1. **Check Health Regularly:** Monitor dashboard status
2. **Review Logs:** Check backend logs for errors
3. **Update Dependencies:** Keep packages up to date
4. **Backup Reports:** Save important analysis reports

### Performance Tips

1. **Image Size:** Optimal size: 1000-2000 pixels
2. **Batch Processing:** Process multiple images sequentially
3. **Network:** Ensure stable internet for API calls
4. **Resources:** Close unnecessary applications

---

## Reproducing Exact Output

To reproduce the exact output shown in the system:

### Step-by-Step Reproduction:

1. **Start both servers** (backend and frontend)
2. **Verify system health** (dashboard shows all green)
3. **Upload test image:** `test_images/wafer_CMP_defects_02.png`
4. **Wait for analysis** (10-30 seconds)
5. **Expected Results:**
   - Total Defects: **9**
   - Severity Score: **~50%** (HIGH)
   - Defect Types: **2** (Pattern Defects, Scratches)
   - Pattern Defects: **8**
   - Scratches: **1**
6. **Download report** - PDF will be generated

### Verification Checklist:

- [ ] Backend running on port 8001
- [ ] Frontend running on port 3002
- [ ] All agents show as ready
- [ ] Models show as loaded (HF DETR, HF ViT)
- [ ] Image uploads successfully
- [ ] Analysis completes without errors
- [ ] Results display correctly
- [ ] Report downloads successfully

---

## Support

For issues or questions:
1. Check backend logs in terminal
2. Review API documentation: `http://localhost:8001/docs`
3. Check this user manual
4. Review error messages for specific issues

---

**Last Updated:** November 2024
**Version:** 1.0.0

