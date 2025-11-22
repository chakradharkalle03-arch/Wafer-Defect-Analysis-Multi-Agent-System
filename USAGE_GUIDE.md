# Usage Guide

## Quick Start

### 1. Start the Backend

```bash
# Windows
start_backend.bat

# Linux/Mac
./start_backend.sh
```

The backend will start on `http://localhost:8000`

### 2. Start the Frontend

```bash
cd frontend
npm install  # First time only
npm start
```

The frontend will start on `http://localhost:3000`

### 3. Using the System

1. **Open the Web Interface**
   - Navigate to `http://localhost:3000`
   - Check the System Dashboard to verify all agents are ready

2. **Upload a Wafer Image**
   - Click or drag-and-drop a wafer image
   - Supported formats: JPG, PNG, TIFF
   - Optionally enter Wafer ID and Batch ID

3. **Wait for Analysis**
   - The system will process the image
   - Progress is shown in real-time
   - Analysis typically takes 30-60 seconds

4. **Review Results**
   - **Overview Tab**: Summary statistics and defect counts
   - **Defects Tab**: Detailed defect information
   - **Root Causes Tab**: Process step analysis and recommendations
   - **Analytics Tab**: Visual charts and graphs

5. **Download Report**
   - Click "Download Full Report (PDF)" button
   - Get comprehensive QC report with all analysis

## API Usage

### Using cURL

```bash
# Analyze an image
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@wafer_image.jpg" \
  -F "wafer_id=WAFER001" \
  -F "batch_id=BATCH001"
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/analyze"
files = {"file": open("wafer_image.jpg", "rb")}
data = {"wafer_id": "WAFER001", "batch_id": "BATCH001"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Total defects: {result['total_defects']}")
print(f"Severity score: {result['severity_score']}")
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('wafer_id', 'WAFER001');

fetch('http://localhost:8000/api/v1/analyze', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Understanding Results

### Severity Score

- **0.0 - 0.3**: LOW - Minor defects, acceptable quality
- **0.3 - 0.5**: MODERATE - Some defects, review recommended
- **0.5 - 0.8**: HIGH - Significant defects, action required
- **0.8 - 1.0**: CRITICAL - Major defects, immediate action needed

### Defect Types

- **CMP Defects**: Chemical Mechanical Polishing issues
- **Litho Hotspots**: Lithography process problems
- **Pattern Bridging**: Unintended connections between patterns
- **Scratches**: Mechanical damage
- **Particles**: Contamination
- **Pattern Defects**: Pattern-related issues
- **Etch Defects**: Etching process problems
- **Deposition Defects**: Deposition process issues

### Process Steps

- **CMP**: Chemical Mechanical Polishing
- **Lithography**: Pattern transfer
- **Etch**: Material removal
- **Deposition**: Material addition
- **Implant**: Ion implantation
- **Cleaning**: Wafer cleaning

## Best Practices

1. **Image Quality**
   - Use high-resolution images
   - Ensure good contrast
   - Avoid compression artifacts

2. **Batch Processing**
   - Use consistent Batch IDs for tracking
   - Maintain Wafer ID conventions

3. **Monitoring**
   - Check system health regularly
   - Monitor model loading status
   - Review error logs

4. **Reports**
   - Download reports for record keeping
   - Use reports for quality documentation
   - Share reports with process engineers

## Troubleshooting

### Analysis Fails

- Check image format (JPG, PNG, TIFF)
- Verify image is not corrupted
- Check backend logs for errors
- Ensure models are loaded

### Slow Performance

- Check system resources (RAM, CPU)
- Enable GPU if available (CUDA)
- Reduce image resolution if needed
- Check network connection for model downloads

### Models Not Loading

- Verify HuggingFace API key
- Check internet connection
- Verify model cache directory permissions
- Check available disk space

## Advanced Usage

### Custom Configuration

Edit `app/core/config.py` to:
- Adjust confidence thresholds
- Add custom defect categories
- Modify process steps
- Change model selections

### Batch Processing

Create a script to process multiple images:

```python
import os
from app.core.orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()
image_dir = "path/to/images"

for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    result = orchestrator.analyze_wafer(image_path)
    print(f"{image_file}: {result.total_defects} defects")
```

### Integration

The system can be integrated with:
- Manufacturing execution systems (MES)
- Quality management systems (QMS)
- Data analytics platforms
- Reporting dashboards

## Support

For issues or questions:
1. Check the logs in the backend console
2. Review the API documentation at `/docs`
3. Check the installation guide
4. Review error messages for specific issues

