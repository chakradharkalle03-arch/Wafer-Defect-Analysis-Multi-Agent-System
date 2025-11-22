"""
Pydantic schemas for request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DefectType(str, Enum):
    """Defect type enumeration"""
    CMP_DEFECTS = "CMP_defects"
    LITHO_HOTSPOTS = "litho_hotspots"
    PATTERN_BRIDGING = "pattern_bridging"
    SCRATCHES = "scratches"
    PARTICLES = "particles"
    PATTERN_DEFECTS = "pattern_defects"
    ETCH_DEFECTS = "etch_defects"
    DEPOSITION_DEFECTS = "deposition_defects"


class ProcessStep(str, Enum):
    """Process step enumeration"""
    CMP = "CMP"
    LITHOGRAPHY = "Lithography"
    ETCH = "Etch"
    DEPOSITION = "Deposition"
    IMPLANT = "Implant"
    CLEANING = "Cleaning"


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x_min: float = Field(..., description="Minimum x coordinate")
    y_min: float = Field(..., description="Minimum y coordinate")
    x_max: float = Field(..., description="Maximum x coordinate")
    y_max: float = Field(..., description="Maximum y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class DefectDetection(BaseModel):
    """Single defect detection result"""
    defect_id: str
    bbox: BoundingBox
    mask: Optional[List[List[int]]] = None  # Segmentation mask
    area: float
    defect_type: Optional[str] = None


class ClassificationResult(BaseModel):
    """Defect classification result"""
    defect_id: str
    defect_type: DefectType
    confidence: float = Field(..., ge=0.0, le=1.0)
    sub_category: Optional[str] = None
    description: Optional[str] = None


class RootCauseAnalysis(BaseModel):
    """Root cause analysis result"""
    defect_id: str
    process_step: ProcessStep
    confidence: float = Field(..., ge=0.0, le=1.0)
    likely_cause: str
    recommendations: List[str]
    historical_similarity: Optional[float] = None


class ImageAnalysisRequest(BaseModel):
    """Request for image analysis"""
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    wafer_id: Optional[str] = None
    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImageAnalysisResponse(BaseModel):
    """Complete image analysis response"""
    analysis_id: str
    wafer_id: Optional[str] = None
    batch_id: Optional[str] = None
    timestamp: datetime
    image_path: str
    
    # Detection results
    defects: List[DefectDetection]
    total_defects: int
    
    # Classification results
    classifications: List[ClassificationResult]
    
    # Root cause analysis
    root_causes: List[RootCauseAnalysis]
    
    # Summary statistics
    defect_summary: Dict[str, int]
    severity_score: float = Field(..., ge=0.0, le=1.0)
    
    # Report
    report_path: Optional[str] = None


class ReportRequest(BaseModel):
    """Request for report generation"""
    analysis_id: str
    include_plots: bool = True
    format: str = "pdf"  # pdf, html, json


class ReportResponse(BaseModel):
    """Report generation response"""
    report_id: str
    report_path: str
    format: str
    generated_at: datetime
    file_size: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents_ready: Dict[str, bool]
    models_loaded: Dict[str, bool]

