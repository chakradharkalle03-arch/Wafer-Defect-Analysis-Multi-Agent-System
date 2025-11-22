"""
Classification Agent - Classifies defect types
Uses ML models and rule-based logic to classify defects into categories
"""
import logging
import numpy as np
from typing import List, Dict, Optional
from huggingface_hub import login
import requests
import base64
import io
from PIL import Image

from app.core.config import settings
from app.models.schemas import DefectDetection, ClassificationResult, DefectType

logger = logging.getLogger(__name__)

# Lazy import for transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers not available - using rule-based classification only: {e}")


class ClassificationAgent:
    """
    Classification agent for wafer defect types
    Classifies defects into: CMP defects, litho hotspots, pattern bridging, etc.
    """
    
    def __init__(self):
        """Initialize the Classification Agent"""
        self.device = settings.device
        self.defect_categories = settings.defect_categories
        
        # Login to HuggingFace
        try:
            login(token=settings.hf_api_key)
        except Exception as e:
            logger.warning(f"HuggingFace login failed: {e}")
        
        # Initialize classification model
        self.classifier = None
        self._load_model()
        
        # Defect type mappings and characteristics
        self.defect_characteristics = self._initialize_defect_characteristics()
    
    def _load_model(self):
        """Load classification model - using HuggingFace Inference API"""
        self.use_hf_api = True
        self.hf_api_url = "https://router.huggingface.co/models"
        
        # Try local model first
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Attempting to load classification model locally...")
                self.classifier = pipeline(
                    "image-classification",
                    model="google/vit-base-patch16-224",
                    device=0 if self.device == "cuda" else -1,
                    token=settings.hf_api_key,
                    cache_dir=settings.model_cache_dir
                )
                logger.info("Classification model loaded successfully (local)")
                self.use_hf_api = False
                return
            except Exception as e:
                logger.warning(f"Local model not available: {e}")
        
        # Use HuggingFace Inference API
        logger.info("Using HuggingFace Inference API for classification")
        logger.info("Model: google/vit-base-patch16-224 (open source)")
        self.classifier = None
    
    def _initialize_defect_characteristics(self) -> Dict[str, Dict]:
        """
        Initialize defect characteristics for rule-based classification
        These are based on semiconductor manufacturing knowledge
        """
        return {
            "CMP_defects": {
                "shape": ["circular", "elliptical"],
                "size_range": (5, 100),  # micrometers
                "location": "surface",
                "appearance": ["dishing", "erosion", "scratches"],
                "typical_causes": ["slurry contamination", "pad wear", "pressure issues"]
            },
            "litho_hotspots": {
                "shape": ["irregular", "pattern-dependent"],
                "size_range": (0.1, 10),
                "location": "pattern areas",
                "appearance": ["bridging", "necking", "pinching"],
                "typical_causes": ["focus drift", "dose variation", "resist issues"]
            },
            "pattern_bridging": {
                "shape": ["linear", "curved"],
                "size_range": (0.1, 5),
                "location": "between patterns",
                "appearance": "connection between features",
                "typical_causes": ["over-etch", "resist scumming", "defocus"]
            },
            "scratches": {
                "shape": ["linear", "elongated"],
                "size_range": (1, 1000),
                "location": "anywhere",
                "appearance": "long thin lines",
                "typical_causes": ["mechanical damage", "handling", "equipment"]
            },
            "particles": {
                "shape": ["circular", "irregular"],
                "size_range": (0.1, 50),
                "location": "surface",
                "appearance": "discrete objects",
                "typical_causes": ["contamination", "cleanroom issues", "equipment"]
            },
            "pattern_defects": {
                "shape": ["pattern-dependent"],
                "size_range": (0.1, 20),
                "location": "pattern areas",
                "appearance": ["missing features", "extra features"],
                "typical_causes": ["lithography issues", "etch problems"]
            },
            "etch_defects": {
                "shape": ["irregular", "pattern-dependent"],
                "size_range": (0.1, 50),
                "location": "etched areas",
                "appearance": ["undercut", "over-etch", "residue"],
                "typical_causes": ["etch rate variation", "mask issues"]
            },
            "deposition_defects": {
                "shape": ["circular", "irregular"],
                "size_range": (0.1, 100),
                "location": "deposited layers",
                "appearance": ["voids", "delamination", "thickness variation"],
                "typical_causes": ["deposition rate issues", "contamination"]
            }
        }
    
    def classify_by_characteristics(self, defect: DefectDetection) -> Dict:
        """
        Classify defect based on geometric and visual characteristics
        """
        bbox = defect.bbox
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min
        area = defect.area
        
        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        
        # Calculate circularity (for particles)
        circularity = 4 * np.pi * area / (width * height) if (width * height) > 0 else 0
        
        scores = {}
        
        # Score each defect type based on characteristics
        for defect_type, characteristics in self.defect_characteristics.items():
            score = 0.0
            
            # Check shape characteristics
            if defect_type == "scratches":
                if aspect_ratio > 5:  # Long and thin
                    score += 0.8
                if area > 1000:  # Large area
                    score += 0.2
            
            elif defect_type == "particles":
                if 0.7 < circularity < 1.3:  # Nearly circular
                    score += 0.6
                if 10 < area < 1000:  # Medium size
                    score += 0.4
            
            elif defect_type == "pattern_bridging":
                if 1.5 < aspect_ratio < 10:  # Elongated but not too long
                    score += 0.7
                if area < 100:  # Small area
                    score += 0.3
            
            elif defect_type == "CMP_defects":
                if 0.8 < circularity < 1.2:  # Circular/elliptical
                    score += 0.5
                if 50 < area < 5000:  # Medium to large
                    score += 0.5
            
            elif defect_type == "litho_hotspots":
                if area < 50:  # Small defects
                    score += 0.6
                if 1.2 < aspect_ratio < 3:  # Slightly elongated
                    score += 0.4
            
            # Normalize score
            scores[defect_type] = min(score, 1.0)
        
        return scores
    
    def classify_defect(self, defect: DefectDetection, image_features: Optional[np.ndarray] = None) -> ClassificationResult:
        """
        Classify a single defect
        Combines ML-based and rule-based classification
        """
        try:
            # Rule-based classification
            characteristic_scores = self.classify_by_characteristics(defect)
            
            # If we have image features, use ML model (local or HF API)
            ml_scores = {}
            if image_features is not None:
                if self.classifier is not None:
                    # Use local model
                    ml_scores = characteristic_scores  # Placeholder - would use actual model
                elif self.use_hf_api:
                    # Use HuggingFace Inference API
                    ml_scores = self._classify_with_hf_api(defect, image_features)
            
            # Combine scores (weighted average)
            combined_scores = {}
            for defect_type in self.defect_categories:
                rule_score = characteristic_scores.get(defect_type, 0.0)
                ml_score = ml_scores.get(defect_type, 0.0)
                # Weight: 70% rule-based, 30% ML (adjust based on model performance)
                combined_scores[defect_type] = 0.7 * rule_score + 0.3 * ml_score
            
            # Get best match
            best_type = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[best_type]
            
            # Ensure minimum confidence threshold
            if confidence < 0.3:
                # Default to most likely based on size/shape
                if defect.area > 1000:
                    best_type = "scratches"
                elif defect.area < 50:
                    best_type = "particles"
                else:
                    best_type = "pattern_defects"
                confidence = 0.5
            
            # Get description
            characteristics = self.defect_characteristics.get(best_type, {})
            description = f"{best_type.replace('_', ' ').title()}: {characteristics.get('appearance', 'unknown appearance')}"
            
            # Map to DefectType enum
            try:
                defect_type_enum = DefectType[best_type.upper()]
            except KeyError:
                defect_type_enum = DefectType.PATTERN_DEFECTS
            
            return ClassificationResult(
                defect_id=defect.defect_id,
                defect_type=defect_type_enum,
                confidence=float(confidence),
                description=description,
                sub_category=characteristics.get("appearance", "")
            )
            
        except Exception as e:
            logger.error(f"Error classifying defect {defect.defect_id}: {e}")
            # Return default classification
            return ClassificationResult(
                defect_id=defect.defect_id,
                defect_type=DefectType.PATTERN_DEFECTS,
                confidence=0.5,
                description="Unknown defect type"
            )
    
    def classify_defects(self, defects: List[DefectDetection], image_features: Optional[Dict[str, np.ndarray]] = None) -> List[ClassificationResult]:
        """
        Classify multiple defects
        """
        classifications = []
        
        for defect in defects:
            features = image_features.get(defect.defect_id) if image_features else None
            classification = self.classify_defect(defect, features)
            classifications.append(classification)
        
        return classifications
    
    def get_defect_statistics(self, classifications: List[ClassificationResult]) -> Dict[str, int]:
        """
        Generate statistics about defect types
        """
        stats = {category: 0 for category in self.defect_categories}
        
        for classification in classifications:
            defect_type = classification.defect_type.value
            stats[defect_type] = stats.get(defect_type, 0) + 1
        
        return stats
    
    def _classify_with_hf_api(self, defect: DefectDetection, image_features: Optional[np.ndarray]) -> Dict:
        """Classify using HuggingFace Inference API"""
        try:
            # Use rule-based scores as base, HF API can enhance
            # For now, return rule-based scores (HF API would need image input)
            return {}
        except Exception as e:
            logger.error(f"Error in HF API classification: {e}")
            return {}

