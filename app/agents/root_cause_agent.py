"""
Root Cause Agent - Infers likely process step causing defect
Uses knowledge base and pattern matching to identify root causes
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from app.core.config import settings
from app.models.schemas import (
    ClassificationResult, 
    RootCauseAnalysis, 
    ProcessStep,
    DefectType
)

logger = logging.getLogger(__name__)


class RootCauseAgent:
    """
    Root Cause Analysis Agent
    Infers likely process step and cause based on defect characteristics
    """
    
    def __init__(self):
        """Initialize the Root Cause Agent"""
        self.process_steps = settings.process_steps
        self.knowledge_base = self._build_knowledge_base()
        self.historical_data = {}  # In production, this would be a database
    
    def _build_knowledge_base(self) -> Dict:
        """
        Build knowledge base mapping defect types to process steps and causes
        Based on semiconductor manufacturing expertise
        """
        return {
            DefectType.CMP_DEFECTS: {
                "primary_process": ProcessStep.CMP,
                "confidence": 0.95,
                "common_causes": [
                    "Slurry contamination or improper composition",
                    "Pad wear or conditioning issues",
                    "Pressure or speed variations",
                    "Post-CMP cleaning inadequacy",
                    "Wafer surface contamination"
                ],
                "recommendations": [
                    "Check slurry quality and composition",
                    "Inspect CMP pad condition and replace if needed",
                    "Review pressure and speed settings",
                    "Verify post-CMP cleaning process",
                    "Check for incoming wafer contamination"
                ]
            },
            DefectType.LITHO_HOTSPOTS: {
                "primary_process": ProcessStep.LITHOGRAPHY,
                "confidence": 0.92,
                "common_causes": [
                    "Focus drift or focus map issues",
                    "Dose variation across wafer",
                    "Resist thickness non-uniformity",
                    "Mask defects or contamination",
                    "Overlay errors"
                ],
                "recommendations": [
                    "Check focus map and recalibrate if needed",
                    "Verify dose uniformity across wafer",
                    "Inspect resist thickness profile",
                    "Review mask quality and cleanliness",
                    "Check overlay registration"
                ]
            },
            DefectType.PATTERN_BRIDGING: {
                "primary_process": ProcessStep.LITHOGRAPHY,
                "confidence": 0.88,
                "common_causes": [
                    "Over-exposure or dose too high",
                    "Defocus conditions",
                    "Resist scumming or residue",
                    "Develop process issues",
                    "Pattern density effects"
                ],
                "recommendations": [
                    "Review exposure dose settings",
                    "Check focus conditions",
                    "Inspect develop process parameters",
                    "Verify resist quality and age",
                    "Consider pattern density compensation"
                ]
            },
            DefectType.SCRATCHES: {
                "primary_process": ProcessStep.CLEANING,
                "confidence": 0.75,
                "common_causes": [
                    "Mechanical handling damage",
                    "Equipment contact during processing",
                    "Improper wafer handling",
                    "Transport system issues",
                    "Equipment malfunction"
                ],
                "recommendations": [
                    "Review wafer handling procedures",
                    "Inspect transport system for damage",
                    "Check equipment for sharp edges",
                    "Verify proper wafer placement",
                    "Review operator training"
                ]
            },
            DefectType.PARTICLES: {
                "primary_process": ProcessStep.CLEANING,
                "confidence": 0.70,
                "common_causes": [
                    "Cleanroom contamination",
                    "Equipment particle generation",
                    "Inadequate cleaning process",
                    "Wafer storage issues",
                    "Personnel contamination"
                ],
                "recommendations": [
                    "Check cleanroom particle counts",
                    "Inspect equipment for particle sources",
                    "Review cleaning process effectiveness",
                    "Verify wafer storage conditions",
                    "Check personnel protocols"
                ]
            },
            DefectType.PATTERN_DEFECTS: {
                "primary_process": ProcessStep.LITHOGRAPHY,
                "confidence": 0.80,
                "common_causes": [
                    "Lithography process variations",
                    "Etch process issues",
                    "Mask defects",
                    "Resist problems",
                    "Process integration issues"
                ],
                "recommendations": [
                    "Review lithography process parameters",
                    "Check etch process conditions",
                    "Inspect mask quality",
                    "Verify resist process",
                    "Review process integration"
                ]
            },
            DefectType.ETCH_DEFECTS: {
                "primary_process": ProcessStep.ETCH,
                "confidence": 0.90,
                "common_causes": [
                    "Etch rate non-uniformity",
                    "Mask erosion or damage",
                    "Chamber contamination",
                    "Gas flow issues",
                    "Temperature variations"
                ],
                "recommendations": [
                    "Check etch rate uniformity",
                    "Inspect mask condition",
                    "Review chamber cleaning schedule",
                    "Verify gas flow rates",
                    "Check temperature control"
                ]
            },
            DefectType.DEPOSITION_DEFECTS: {
                "primary_process": ProcessStep.DEPOSITION,
                "confidence": 0.90,
                "common_causes": [
                    "Deposition rate variations",
                    "Chamber contamination",
                    "Temperature non-uniformity",
                    "Gas flow issues",
                    "Pre-clean inadequacy"
                ],
                "recommendations": [
                    "Check deposition rate uniformity",
                    "Review chamber cleaning",
                    "Verify temperature profile",
                    "Check gas flow rates",
                    "Review pre-clean process"
                ]
            }
        }
    
    def analyze_root_cause(
        self, 
        classification: ClassificationResult,
        defect_count: Optional[int] = None,
        location: Optional[str] = None
    ) -> RootCauseAnalysis:
        """
        Analyze root cause for a single defect classification
        """
        try:
            defect_type = classification.defect_type
            
            # Get knowledge base entry
            kb_entry = self.knowledge_base.get(defect_type, {})
            
            if not kb_entry:
                # Default analysis for unknown types
                return RootCauseAnalysis(
                    defect_id=classification.defect_id,
                    process_step=ProcessStep.CLEANING,
                    confidence=0.5,
                    likely_cause="Unknown defect type - requires investigation",
                    recommendations=["Review defect characteristics", "Check process logs", "Consult process engineer"]
                )
            
            # Base confidence from knowledge base
            base_confidence = kb_entry.get("confidence", 0.7)
            
            # Adjust confidence based on defect count and location
            adjusted_confidence = self._adjust_confidence(
                base_confidence,
                classification.confidence,
                defect_count
            )
            
            # Select most likely cause
            common_causes = kb_entry.get("common_causes", [])
            likely_cause = common_causes[0] if common_causes else "Process variation"
            
            # Get recommendations
            recommendations = kb_entry.get("recommendations", [])
            
            # Add location-specific recommendations if available
            if location:
                recommendations.append(f"Check {location} area specifically")
            
            # Calculate historical similarity (placeholder - would use actual historical data)
            historical_similarity = self._calculate_historical_similarity(
                defect_type,
                defect_count
            )
            
            return RootCauseAnalysis(
                defect_id=classification.defect_id,
                process_step=kb_entry.get("primary_process", ProcessStep.CLEANING),
                confidence=float(adjusted_confidence),
                likely_cause=likely_cause,
                recommendations=recommendations,
                historical_similarity=historical_similarity
            )
            
        except Exception as e:
            logger.error(f"Error analyzing root cause for {classification.defect_id}: {e}")
            return RootCauseAnalysis(
                defect_id=classification.defect_id,
                process_step=ProcessStep.CLEANING,
                confidence=0.5,
                likely_cause="Analysis error - manual review required",
                recommendations=["Review defect manually", "Check system logs"]
            )
    
    def _adjust_confidence(
        self, 
        base_confidence: float,
        classification_confidence: float,
        defect_count: Optional[int]
    ) -> float:
        """
        Adjust confidence based on additional factors
        """
        # Weight base confidence and classification confidence
        adjusted = 0.6 * base_confidence + 0.4 * classification_confidence
        
        # Adjust based on defect count (more defects = higher confidence in process issue)
        if defect_count:
            if defect_count > 10:
                adjusted = min(adjusted + 0.1, 1.0)
            elif defect_count < 3:
                adjusted = max(adjusted - 0.1, 0.3)
        
        return adjusted
    
    def _calculate_historical_similarity(
        self,
        defect_type: DefectType,
        defect_count: Optional[int]
    ) -> Optional[float]:
        """
        Calculate similarity to historical cases
        In production, this would query a database of historical defects
        """
        # Placeholder implementation
        # Would compare against historical data
        if defect_type in self.historical_data:
            return 0.75  # Example similarity score
        return None
    
    def analyze_batch(
        self,
        classifications: List[ClassificationResult],
        total_defects: int
    ) -> List[RootCauseAnalysis]:
        """
        Analyze root causes for a batch of defects
        """
        analyses = []
        
        for classification in classifications:
            analysis = self.analyze_root_cause(
                classification,
                defect_count=total_defects
            )
            analyses.append(analysis)
        
        return analyses
    
    def get_process_summary(self, analyses: List[RootCauseAnalysis]) -> Dict[str, Dict]:
        """
        Generate summary of process issues
        """
        process_summary = {}
        
        for analysis in analyses:
            process = analysis.process_step.value
            
            if process not in process_summary:
                process_summary[process] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "common_causes": []
                }
            
            process_summary[process]["count"] += 1
            process_summary[process]["avg_confidence"] += analysis.confidence
            process_summary[process]["common_causes"].append(analysis.likely_cause)
        
        # Calculate averages
        for process in process_summary:
            count = process_summary[process]["count"]
            if count > 0:
                process_summary[process]["avg_confidence"] /= count
                # Get most common cause
                causes = process_summary[process]["common_causes"]
                most_common = max(set(causes), key=causes.count) if causes else "Unknown"
                process_summary[process]["primary_cause"] = most_common
        
        return process_summary

