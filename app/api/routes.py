"""
FastAPI routes for wafer defect analysis
"""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pathlib import Path
import shutil
import uuid
from datetime import datetime

from app.core.orchestrator import MultiAgentOrchestrator
from app.models.schemas import (
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    ReportRequest,
    ReportResponse,
    HealthResponse
)
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize orchestrator (singleton)
orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = MultiAgentOrchestrator()
    return orchestrator


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        orch = get_orchestrator()
        
        # Get agent status from supervisor
        supervisor_status = orch.supervisor.get_agent_status()
        
        agents_ready = {
            "supervisor_agent": orch.supervisor is not None,
            "image_agent": orch.image_agent is not None,
            "classification_agent": orch.classification_agent is not None,
            "mapping_agent": orch.mapping_agent is not None,
            "root_cause_agent": orch.root_cause_agent is not None,
            "report_agent": orch.report_agent is not None
        }
        
        # Check model availability - using HuggingFace Inference API only
        # YOLO and local ViT removed - using HuggingFace API exclusively
        models_loaded = {
            "huggingface_detr": True,  # DETR model via HF API
            "huggingface_vit": True    # ViT model via HF API
        }
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            agents_ready=agents_ready,
            models_loaded=models_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            agents_ready={},
            models_loaded={}
        )


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_wafer_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    wafer_id: Optional[str] = None,
    batch_id: Optional[str] = None
):
    """
    Analyze wafer image for defects
    Upload an image file and get comprehensive defect analysis
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {settings.allowed_extensions}"
            )
        
        # Save uploaded file
        upload_dir = Path("data/raw")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}{file_ext}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Run analysis
        orch = get_orchestrator()
        analysis = orch.analyze_wafer(
            image_path=str(file_path),
            wafer_id=wafer_id,
            batch_id=batch_id,
            metadata={"original_filename": file.filename}
        )
        
        # Cleanup old files in background
        background_tasks.add_task(cleanup_old_files, upload_dir)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing wafer image: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-url", response_model=ImageAnalysisResponse)
async def analyze_wafer_from_url(
    request: ImageAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze wafer image from URL
    """
    try:
        if not request.image_url:
            raise HTTPException(status_code=400, detail="No image URL provided")
        
        # Download image
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(request.image_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download image")
            
            # Save image
            upload_dir = Path("data/raw")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_id = str(uuid.uuid4())
            file_ext = Path(request.image_url).suffix or ".jpg"
            file_path = upload_dir / f"{file_id}{file_ext}"
            
            with open(file_path, "wb") as f:
                f.write(response.content)
        
        # Run analysis
        orch = get_orchestrator()
        analysis = orch.analyze_wafer(
            image_path=str(file_path),
            wafer_id=request.wafer_id,
            batch_id=request.batch_id,
            metadata=request.metadata or {}
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing wafer from URL: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/report/{analysis_id}")
async def get_report(analysis_id: str):
    """
    Download generated report
    """
    try:
        reports_dir = Path("reports")
        # Find report file
        report_files = list(reports_dir.glob(f"*{analysis_id}*.pdf"))
        
        if not report_files:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_path = report_files[0]
        
        return FileResponse(
            path=str(report_path),
            filename=report_path.name,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


@router.get("/analysis/{analysis_id}", response_model=ImageAnalysisResponse)
async def get_analysis(analysis_id: str):
    """
    Get analysis results by ID
    Note: In production, this would query a database
    """
    # Placeholder - in production, store analyses in database
    raise HTTPException(status_code=501, detail="Analysis retrieval not yet implemented")


@router.get("/map/{map_filename}")
async def get_map_image(map_filename: str):
    """
    Serve defect map images
    """
    try:
        map_path = Path("reports/plots") / map_filename
        
        if not map_path.exists():
            raise HTTPException(status_code=404, detail="Map image not found")
        
        return FileResponse(
            path=str(map_path),
            filename=map_filename,
            media_type="image/png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving map image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve map image: {str(e)}")


def cleanup_old_files(directory: Path, max_age_days: int = 7):
    """Clean up old files"""
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

