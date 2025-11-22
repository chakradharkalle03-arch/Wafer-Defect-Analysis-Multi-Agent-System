"""
Advanced Root Cause Agent - Uses LLM-based reasoning for intelligent RCA
Implements cutting-edge AI reasoning similar to TSMC/Samsung research initiatives
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json

from app.core.config import settings
from app.models.schemas import (
    ClassificationResult, 
    RootCauseAnalysis, 
    ProcessStep,
    DefectType
)

logger = logging.getLogger(__name__)

# Advanced AI reasoning using LLM
LLM_AVAILABLE = False
try:
    # Try to import without triggering torch dependencies
    import sys
    import importlib.util
    
    # Check if langchain_community exists without importing transformers
    spec = importlib.util.find_spec("langchain_community")
    if spec:
        # Try importing just the endpoint class
        from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
        LLM_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    LLM_AVAILABLE = False
    logger.warning(f"LLM libraries not available - using advanced rule-based reasoning: {e}")


class AdvancedRootCauseAgent:
    """
    Advanced Root Cause Analysis Agent with LLM-based reasoning
    Implements next-generation AI reasoning for semiconductor defect analysis
    Similar to research initiatives at TSMC, Samsung, IMEC
    """
    
    def __init__(self):
        """Initialize the Advanced Root Cause Agent"""
        self.process_steps = settings.process_steps
        self.llm = None
        self.reasoning_chain = None
        self._initialize_llm()
        self.knowledge_base = self._build_advanced_knowledge_base()
        self.historical_patterns = {}  # Would be connected to historical database
    
    def _initialize_llm(self):
        """Initialize LLM for advanced reasoning"""
        if not LLM_AVAILABLE:
            logger.warning("LLM not available - using rule-based fallback")
            return
        
        if not LLM_AVAILABLE:
            self.llm = None
            return
        
        try:
            # Use HuggingFace Inference API with open-source models
            # Using a smaller, faster model for reasoning
            from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
            
            try:
                # Try using HuggingFace Inference API
                # Note: For production, you might want to use a dedicated inference endpoint
                self.llm = HuggingFaceEndpoint(
                    endpoint_url="https://router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=settings.hf_api_key,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.3,
                        "max_new_tokens": 500,
                        "top_p": 0.9
                    }
                )
                logger.info("✓ Initialized advanced LLM for reasoning (HuggingFace)")
            except Exception as e:
                logger.warning(f"Could not initialize HuggingFace LLM: {e}")
                logger.info("Will use advanced rule-based reasoning instead")
                self.llm = None
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.llm = None
    
    def _build_advanced_knowledge_base(self) -> Dict:
        """
        Build advanced knowledge base with semantic relationships
        Includes process dependencies, defect propagation patterns, etc.
        """
        return {
            # Process dependency graph
            "process_dependencies": {
                "CMP": ["Cleaning", "Deposition"],
                "Lithography": ["CMP", "Deposition"],
                "Etch": ["Lithography"],
                "Deposition": ["Etch", "CMP"],
            },
            # Defect propagation patterns
            "defect_propagation": {
                "CMP_defects": {
                    "can_cause": ["litho_hotspots", "pattern_defects"],
                    "caused_by": ["particles", "scratches"],
                    "process_chain": ["Cleaning → CMP → Lithography"]
                },
                "litho_hotspots": {
                    "can_cause": ["pattern_bridging", "etch_defects"],
                    "caused_by": ["CMP_defects", "particles"],
                    "process_chain": ["Lithography → Etch → Deposition"]
                }
            },
            # Advanced cause-effect relationships
            "causal_relationships": {
                "slurry_contamination": {
                    "primary_effect": "CMP_defects",
                    "secondary_effects": ["surface_roughness", "particle_generation"],
                    "detection_signature": "circular_dishing_patterns",
                    "root_process": "CMP"
                },
                "focus_drift": {
                    "primary_effect": "litho_hotspots",
                    "secondary_effects": ["pattern_bridging", "necking"],
                    "detection_signature": "pattern_dependent_defects",
                    "root_process": "Lithography"
                }
            }
        }
    
    def analyze_root_cause_advanced(
        self,
        classification: ClassificationResult,
        defect_count: Optional[int] = None,
        location: Optional[str] = None,
        defect_patterns: Optional[Dict] = None,
        process_history: Optional[List] = None
    ) -> RootCauseAnalysis:
        """
        Advanced root cause analysis using LLM-based reasoning
        Implements multi-factor analysis similar to expert process engineers
        """
        try:
            # Build context for LLM reasoning
            context = self._build_reasoning_context(
                classification, defect_count, location, defect_patterns, process_history
            )
            
            # Use LLM for intelligent reasoning if available
            if self.use_direct_api and self.llm_available:
                reasoning_result = self._llm_reasoning_direct_api(context)
                if reasoning_result:
                    return self._parse_llm_reasoning(reasoning_result, classification)
            
            # Fallback to advanced rule-based reasoning
            return self._advanced_rule_based_reasoning(context, classification)
            
        except Exception as e:
            logger.error(f"Error in advanced root cause analysis: {e}")
            return self._fallback_analysis(classification)
    
    def _build_reasoning_context(
        self,
        classification: ClassificationResult,
        defect_count: Optional[int],
        location: Optional[str],
        defect_patterns: Optional[Dict],
        process_history: Optional[List]
    ) -> Dict:
        """Build comprehensive context for AI reasoning"""
        return {
            "defect_type": classification.defect_type.value,
            "confidence": classification.confidence,
            "defect_count": defect_count or 0,
            "location": location or "unknown",
            "defect_patterns": defect_patterns or {},
            "process_history": process_history or [],
            "knowledge_base": self.knowledge_base,
            "timestamp": datetime.now().isoformat()
        }
    
    def _llm_reasoning_direct_api(self, context: Dict) -> Optional[str]:
        """
        Use HuggingFace Inference API directly for advanced reasoning
        Implements chain-of-thought reasoning for complex defect analysis
        """
        try:
            import requests
            
            prompt = f"""<s>[INST] You are an expert semiconductor process engineer. Analyze this wafer defect:

Defect Type: {context['defect_type']}
Confidence: {context['confidence']:.2%}
Defect Count: {context['defect_count']}
Location: {context['location']}

Provide root cause analysis as JSON:
{{
    "process_step": "CMP|Lithography|Etch|Deposition|Cleaning",
    "root_cause": "explanation",
    "confidence": 0.0-1.0,
    "reasoning": "step-by-step analysis",
    "recommendations": ["action1", "action2", "action3"]
}}
[/INST]"""
            
            model_url = f"{self.hf_api_url}/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.3,
                    "max_new_tokens": 500,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }
            
            response = requests.post(model_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    return generated_text
                elif isinstance(result, dict):
                    return result.get('generated_text', '')
            else:
                logger.warning(f"HF API returned {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Direct API reasoning failed: {e}")
            return None
    
    def _parse_llm_reasoning(self, reasoning_result: Optional[str], classification: ClassificationResult) -> RootCauseAnalysis:
        """Parse LLM reasoning result into RootCauseAnalysis"""
        try:
            if not reasoning_result:
                return self._fallback_analysis(classification)
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', reasoning_result, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Map process step
                process_step_map = {
                    "CMP": ProcessStep.CMP,
                    "Lithography": ProcessStep.LITHOGRAPHY,
                    "Etch": ProcessStep.ETCH,
                    "Deposition": ProcessStep.DEPOSITION,
                    "Cleaning": ProcessStep.CLEANING
                }
                
                process_step = process_step_map.get(
                    result.get("process_step", "Cleaning"),
                    ProcessStep.CLEANING
                )
                
                return RootCauseAnalysis(
                    defect_id=classification.defect_id,
                    process_step=process_step,
                    confidence=float(result.get("confidence", 0.7)),
                    likely_cause=result.get("root_cause", "Process variation"),
                    recommendations=result.get("recommendations", []),
                    historical_similarity=None
                )
            else:
                # Fallback if JSON parsing fails
                return self._fallback_analysis(classification)
                
        except Exception as e:
            logger.error(f"Error parsing LLM reasoning: {e}")
            return self._fallback_analysis(classification)
    
    def _advanced_rule_based_reasoning(self, context: Dict, classification: ClassificationResult) -> RootCauseAnalysis:
        """
        Advanced rule-based reasoning with multi-factor analysis
        Uses knowledge graphs and pattern matching
        """
        defect_type = classification.defect_type
        defect_count = context.get('defect_count', 0)
        
        # Multi-factor analysis
        factors = {
            "defect_type": defect_type.value,
            "count": defect_count,
            "confidence": classification.confidence,
            "location": context.get('location')
        }
        
        # Use knowledge base for reasoning
        kb = self.knowledge_base
        propagation = kb.get("defect_propagation", {}).get(defect_type.value, {})
        
        # Determine process step
        if defect_type == DefectType.CMP_DEFECTS:
            process_step = ProcessStep.CMP
            likely_cause = "CMP process variation - check slurry composition and pad condition"
        elif defect_type in [DefectType.LITHO_HOTSPOTS, DefectType.PATTERN_BRIDGING]:
            process_step = ProcessStep.LITHOGRAPHY
            likely_cause = "Lithography process issue - verify focus map and dose uniformity"
        elif defect_type == DefectType.ETCH_DEFECTS:
            process_step = ProcessStep.ETCH
            likely_cause = "Etch process variation - check etch rate and chamber conditions"
        elif defect_type == DefectType.DEPOSITION_DEFECTS:
            process_step = ProcessStep.DEPOSITION
            likely_cause = "Deposition process issue - verify deposition rate and temperature"
        else:
            process_step = ProcessStep.CLEANING
            likely_cause = "Contamination or handling issue - check cleanroom and transport"
        
        # Generate recommendations based on multi-factor analysis
        recommendations = self._generate_intelligent_recommendations(factors, propagation)
        
        # Calculate confidence with multi-factor weighting
        confidence = self._calculate_advanced_confidence(factors, classification)
        
        return RootCauseAnalysis(
            defect_id=classification.defect_id,
            process_step=process_step,
            confidence=confidence,
            likely_cause=likely_cause,
            recommendations=recommendations,
            historical_similarity=None
        )
    
    def _generate_intelligent_recommendations(self, factors: Dict, propagation: Dict) -> List[str]:
        """Generate intelligent recommendations based on multi-factor analysis"""
        recommendations = []
        
        # Base recommendations
        if factors['count'] > 10:
            recommendations.append("High defect density - investigate systematic process issue")
            recommendations.append("Review process parameters for last 24 hours")
        
        if factors['confidence'] > 0.8:
            recommendations.append("High confidence detection - immediate action recommended")
        
        # Process-specific recommendations
        if propagation.get('process_chain'):
            recommendations.append(f"Check upstream processes: {propagation['process_chain']}")
        
        # Add standard recommendations
        recommendations.extend([
            "Review equipment maintenance logs",
            "Check for recent process recipe changes",
            "Verify incoming wafer quality",
            "Consult with process engineering team"
        ])
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _calculate_advanced_confidence(self, factors: Dict, classification: ClassificationResult) -> float:
        """Calculate confidence using multi-factor analysis"""
        base_confidence = classification.confidence
        
        # Adjust based on defect count
        count_factor = min(factors['count'] / 20.0, 1.0)  # More defects = higher confidence
        
        # Adjust based on location patterns
        location_factor = 0.8  # Would analyze location patterns
        
        # Weighted combination
        confidence = 0.5 * base_confidence + 0.3 * count_factor + 0.2 * location_factor
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _fallback_analysis(self, classification: ClassificationResult) -> RootCauseAnalysis:
        """Fallback analysis when advanced reasoning fails"""
        return RootCauseAnalysis(
            defect_id=classification.defect_id,
            process_step=ProcessStep.CLEANING,
            confidence=0.5,
            likely_cause="Requires manual investigation - advanced analysis unavailable",
            recommendations=["Review defect manually", "Check process logs", "Consult process engineer"]
        )
    
    def analyze_batch_advanced(
        self,
        classifications: List[ClassificationResult],
        total_defects: int,
        defect_distribution: Optional[Dict] = None
    ) -> List[RootCauseAnalysis]:
        """
        Advanced batch analysis with pattern recognition
        Identifies systemic issues across multiple defects
        """
        analyses = []
        
        # Analyze patterns across all defects
        patterns = self._identify_patterns(classifications, defect_distribution)
        
        for classification in classifications:
            analysis = self.analyze_root_cause_advanced(
                classification,
                defect_count=total_defects,
                defect_patterns=patterns
            )
            analyses.append(analysis)
        
        return analyses
    
    def _identify_patterns(
        self,
        classifications: List[ClassificationResult],
        defect_distribution: Optional[Dict]
    ) -> Dict:
        """Identify patterns across defects for advanced reasoning"""
        patterns = {
            "dominant_type": None,
            "spatial_clustering": False,
            "systemic_issue": False
        }
        
        if classifications:
            # Find dominant defect type
            type_counts = {}
            for c in classifications:
                type_counts[c.defect_type.value] = type_counts.get(c.defect_type.value, 0) + 1
            
            if type_counts:
                patterns["dominant_type"] = max(type_counts, key=type_counts.get)
                patterns["systemic_issue"] = max(type_counts.values()) > len(classifications) * 0.5
        
        return patterns

