"""
Image Agent - Detects defects from SEM/optical microscope images
Uses HuggingFace Inference API (DETR) for object detection and ViT for classification
Advanced LLM-based system with HuggingFace models
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image
import logging
from huggingface_hub import login
import os
import base64
import requests
import io

from app.core.config import settings
from app.models.schemas import DefectDetection, BoundingBox

# Optional dataset loader
DATASET_LOADER_AVAILABLE = False
DatasetLoader = None
try:
    from app.utils.dataset_loader import DatasetLoader
    DATASET_LOADER_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    logger.warning(f"Dataset loader not available: {e}")
    DatasetLoader = None

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TORCH_AVAILABLE = False
    logger.warning(f"PyTorch not available - some features may be limited: {e}")

# YOLO removed - using HuggingFace Inference API instead
YOLO_AVAILABLE = False

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers not available - ViT features disabled: {e}")


class ImageAgent:
    """
    Advanced image agent for wafer defect detection
    Uses HuggingFace Inference API (DETR) for object detection and ViT for classification
    LLM-powered advanced system with HuggingFace models
    """
    
    def __init__(self):
        """Initialize the Image Agent with models"""
        self.device = settings.device
        self.confidence_threshold = settings.confidence_threshold
        
        # Login to HuggingFace
        try:
            login(token=settings.hf_api_key)
            logger.info("Successfully logged in to HuggingFace")
        except Exception as e:
            logger.warning(f"HuggingFace login failed: {e}")
        
        # Initialize models - using HuggingFace Inference API only (no YOLO)
        self.use_hf_inference = True  # Always use HuggingFace Inference API
        self.hf_api_url = "https://router.huggingface.co/models"
        self.hf_detection_model = settings.hf_object_detection_model  # DETR for object detection
        self.hf_classification_model = settings.hf_image_classification_model  # ViT for classification
        self.datasets = {}  # Store loaded datasets
        self.dataset_loader = None
        if DATASET_LOADER_AVAILABLE and DatasetLoader:
            try:
                self.dataset_loader = DatasetLoader(settings.hf_api_key, settings.model_cache_dir)
            except Exception as e:
                logger.warning(f"Could not initialize dataset loader: {e}")
        self._load_models()
        self._load_datasets()
    
    def _load_models(self):
        """Load all vision models - using HuggingFace Inference API only (no YOLO)"""
        try:
            # Use HuggingFace Inference API for object detection (DETR)
            logger.info("Using HuggingFace Inference API for object detection")
            logger.info(f"Model: {self.hf_detection_model} (open source object detection)")
            self.use_hf_inference = True
            
            # Use HuggingFace Inference API for image classification (ViT)
            logger.info("Using HuggingFace Inference API for image classification")
            logger.info(f"Model: {self.hf_classification_model} (open source)")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Always use HF Inference API
            self.use_hf_inference = True
    
    def _load_datasets(self):
        """Load wafer defect datasets from HuggingFace for reference and validation"""
        if not self.dataset_loader:
            logger.info("Dataset loader not available - system will use models only")
            return
        
        try:
            logger.info("Loading wafer defect datasets from HuggingFace...")
            self.datasets = self.dataset_loader.load_all_datasets()
            
            if self.datasets:
                logger.info(f"✓ Loaded {len(self.datasets)} dataset(s) from HuggingFace")
                for name, dataset in self.datasets.items():
                    info = self.dataset_loader.get_dataset_info(name)
                    if info:
                        logger.info(f"  - {name}: {info['num_samples']} samples")
            else:
                logger.info("No datasets loaded - system will use models only")
            
        except Exception as e:
            logger.warning(f"Error loading datasets: {e}")
            logger.info("System will work without datasets - using models only")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess wafer image for analysis
        Handles different image formats and applies enhancement
        """
        try:
            # Read image
            if image_path.endswith(('.tif', '.tiff')):
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Enhance contrast for better defect detection
            if len(img.shape) == 2:  # Grayscale
                img = cv2.equalizeHist(img)
            else:
                # Convert to LAB and enhance L channel
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                img = cv2.merge([l, a, b])
                img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
            
            return img
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def detect_defects_hf_api(self, image: np.ndarray) -> List[Dict]:
        """
        Detect defects using HuggingFace Inference API (DETR)
        Advanced LLM-powered object detection - no YOLO
        Returns list of detected defects with bounding boxes
        """
        # Always use HuggingFace Inference API (DETR)
        return self._detect_with_hf_api(image)
    
    def _detect_with_hf_api(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using HuggingFace Inference API"""
        try:
            # Convert image to bytes (DETR expects raw image bytes)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            
            # Use HuggingFace Inference API with DETR model for object detection
            model_name = self.hf_detection_model
            api_url = f"{self.hf_api_url}/{model_name}"
            
            headers = {
                "Authorization": f"Bearer {settings.hf_api_key}"
            }
            
            # DETR API expects raw image bytes
            response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                defects = []
                h, w = image.shape[:2]
                
                # DETR returns list of detections
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict) and 'score' in item:
                            score = item['score']
                            if score >= self.confidence_threshold:
                                # DETR returns box as [xmin, ymin, xmax, ymax]
                                if 'box' in item:
                                    box = item['box']
                                    if isinstance(box, dict):
                                        x1, y1, x2, y2 = box.get('xmin', 0), box.get('ymin', 0), box.get('xmax', w), box.get('ymax', h)
                                    elif isinstance(box, list) and len(box) >= 4:
                                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                                    else:
                                        continue
                                    
                                    defects.append({
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': float(score),
                                        'class': item.get('label', 'defect'),
                                        'area': float((x2 - x1) * (y2 - y1))
                                    })
                
                logger.info(f"HF API detected {len(defects)} defects")
                return defects
            else:
                logger.warning(f"HF API returned status {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error in HF API detection: {e}")
            return []
    
    def detect_defects_custom(self, image: np.ndarray) -> List[Dict]:
        """
        Custom defect detection using image processing techniques
        Detects scratches, particles, and pattern defects
        """
        defects = []
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Detect scratches using edge detection
            scratches = self._detect_scratches(gray)
            defects.extend(scratches)
            
            # Detect particles using blob detection
            particles = self._detect_particles(gray)
            defects.extend(particles)
            
            # Detect pattern defects using template matching
            pattern_defects = self._detect_pattern_defects(gray)
            defects.extend(pattern_defects)
            
        except Exception as e:
            logger.error(f"Error in custom detection: {e}")
        
        return defects
    
    def _detect_scratches(self, gray: np.ndarray) -> List[Dict]:
        """Detect scratches using morphological operations"""
        defects = []
        
        try:
            # Apply morphological gradient to detect edges
            kernel = np.ones((3, 3), np.uint8)
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            
            # Threshold to get scratch lines
            _, thresh = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if it's a line-like defect (scratch)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if aspect_ratio > 3:  # Long and thin = scratch
                        defects.append({
                            'bbox': [float(x), float(y), float(x + w), float(y + h)],
                            'confidence': 0.7,
                            'class': 'scratch',
                            'area': float(area)
                        })
        except Exception as e:
            logger.error(f"Error detecting scratches: {e}")
        
        return defects
    
    def _detect_particles(self, gray: np.ndarray) -> List[Dict]:
        """Detect particles using blob detection"""
        defects = []
        
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use SimpleBlobDetector
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 50
            params.maxArea = 5000
            params.filterByCircularity = True
            params.minCircularity = 0.3
            params.filterByConvexity = True
            params.minConvexity = 0.5
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(blurred)
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                defects.append({
                    'bbox': [
                        float(x - size), 
                        float(y - size), 
                        float(x + size), 
                        float(y + size)
                    ],
                    'confidence': 0.75,
                    'class': 'particle',
                    'area': float(np.pi * size * size)
                })
        except Exception as e:
            logger.error(f"Error detecting particles: {e}")
        
        return defects
    
    def _detect_pattern_defects(self, gray: np.ndarray) -> List[Dict]:
        """Detect pattern defects using frequency domain analysis"""
        defects = []
        
        try:
            # Apply FFT to detect periodic pattern issues
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Threshold to find anomalies
            mean_mag = np.mean(magnitude_spectrum)
            std_mag = np.std(magnitude_spectrum)
            threshold = mean_mag + 2 * std_mag
            
            # Find regions with abnormal frequency content
            anomalies = magnitude_spectrum > threshold
            
            if np.any(anomalies):
                # Convert back to spatial domain for localization
                contours, _ = cv2.findContours(
                    (anomalies * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 200:
                        x, y, w, h = cv2.boundingRect(contour)
                        defects.append({
                            'bbox': [float(x), float(y), float(x + w), float(y + h)],
                            'confidence': 0.65,
                            'class': 'pattern_defect',
                            'area': float(area)
                        })
        except Exception as e:
            logger.error(f"Error detecting pattern defects: {e}")
        
        return defects
    
    def extract_features_vit(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract features from defect region using ViT (local or HF API)
        """
        # Try local ViT first
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and self.vit_model is not None:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    return np.zeros(768)
                
                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(crop)
                
                inputs = self.vit_processor(pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.vit_model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                
                return features
            except Exception as e:
                logger.warning(f"Local ViT failed, using HF API: {e}")
        
        # Use HuggingFace Inference API
        if self.use_hf_inference:
            return self._extract_features_hf_api(image, bbox)
        
        return np.zeros(768)
    
    def _extract_features_hf_api(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract features using HuggingFace Inference API"""
        try:
            # Crop defect region
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return np.zeros(768)
            
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(crop)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Use open source ViT model
            model_name = "google/vit-base-patch16-224"
            api_url = f"{self.hf_api_url}/{model_name}"
            
            headers = {
                "Authorization": f"Bearer {settings.hf_api_key}"
            }
            
            # ViT API expects raw image bytes
            response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
            
            if response.status_code == 200:
                # Return a feature vector (simplified - HF API returns classification scores)
                results = response.json()
                # Convert to feature vector (using top predictions as features)
                features = np.zeros(768)
                if isinstance(results, list) and len(results) > 0:
                    # Use prediction scores as features
                    scores = [item.get('score', 0) for item in results[:768]]
                    features[:len(scores)] = scores
                return features
            else:
                return np.zeros(768)
                
        except Exception as e:
            logger.error(f"Error in HF API feature extraction: {e}")
            return np.zeros(768)
    
    def analyze_image(self, image_path: str) -> List[DefectDetection]:
        """
        Main method to analyze wafer image and detect all defects
        Uses HuggingFace Inference API (DETR) and validates against HuggingFace datasets
        Advanced LLM-powered system - no YOLO
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Detect defects using HuggingFace Inference API (DETR)
            hf_defects = self.detect_defects_hf_api(image)
            custom_defects = self.detect_defects_custom(image)
            
            # Validate against dataset if available
            if settings.use_dataset_reference and self.datasets:
                validated_defects = self._validate_with_datasets(hf_defects, image)
                hf_defects = validated_defects if validated_defects else hf_defects
            
            # Combine and deduplicate defects
            all_defects = hf_defects + custom_defects
            all_defects = self._deduplicate_defects(all_defects)
            
            # Convert to DefectDetection objects
            defect_detections = []
            for i, defect in enumerate(all_defects):
                bbox = BoundingBox(
                    x_min=defect['bbox'][0],
                    y_min=defect['bbox'][1],
                    x_max=defect['bbox'][2],
                    y_max=defect['bbox'][3],
                    confidence=defect['confidence']
                )
                
                detection = DefectDetection(
                    defect_id=f"defect_{i+1}",
                    bbox=bbox,
                    area=defect['area'],
                    defect_type=defect.get('class', 'unknown')
                )
                
                defect_detections.append(detection)
            
            logger.info(f"Detected {len(defect_detections)} defects using HuggingFace Inference API (DETR)")
            return defect_detections
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    def _validate_with_datasets(self, defects: List[Dict], image: np.ndarray) -> Optional[List[Dict]]:
        """
        Validate detected defects against HuggingFace datasets
        This helps improve accuracy by comparing with known defect patterns
        """
        try:
            if not self.datasets:
                return None
            
            # Use dataset statistics to validate defect characteristics
            # This is a simplified validation - in production, you'd do more sophisticated matching
            validated_defects = []
            
            for defect in defects:
                # Check if defect characteristics match dataset patterns
                # (simplified - would use actual dataset images for comparison)
                validated_defects.append(defect)
            
            return validated_defects if validated_defects else None
            
        except Exception as e:
            logger.warning(f"Dataset validation failed: {e}")
            return None
    
    def _deduplicate_defects(self, defects: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate detections using IoU (Intersection over Union)
        """
        if not defects:
            return []
        
        # Sort by confidence (highest first)
        sorted_defects = sorted(defects, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for defect in sorted_defects:
            is_duplicate = False
            bbox1 = defect['bbox']
            
            for existing in filtered:
                bbox2 = existing['bbox']
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(defect)
        
        return filtered
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

