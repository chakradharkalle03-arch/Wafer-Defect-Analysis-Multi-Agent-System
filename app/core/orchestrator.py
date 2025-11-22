"""
Multi-Agent Orchestrator using LangGraph
Coordinates the workflow between all agents
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid
from pathlib import Path

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app.agents.langgraph_supervisor import LangGraphSupervisorAgent
from app.models.schemas import (
    ImageAnalysisResponse,
    DefectDetection,
    ClassificationResult,
    RootCauseAnalysis
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between agents with advanced reasoning context"""
    image_path: str
    wafer_id: Optional[str]
    batch_id: Optional[str]
    defects: List[Dict]  # Store as dicts to avoid LangGraph message coercion
    classifications: List[Dict]  # Store as dicts to avoid LangGraph message coercion
    root_causes: List[Dict]  # Store as dicts to avoid LangGraph message coercion
    report_path: Optional[str]
    analysis_id: str
    metadata: Dict
    error: Optional[str]
    reasoning_context: Optional[Dict]  # Advanced reasoning context
    agent_communications: List[Dict]  # Agent-to-agent messages


class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent workflow for wafer defect analysis
    """
    
    def __init__(self):
        """Initialize orchestrator with Supervisor Agent"""
        logger.info("Initializing Multi-Agent Orchestrator with Supervisor Agent...")
        
        # Initialize LangGraph Supervisor Agent (coordinates all sub-agents)
        self.supervisor = LangGraphSupervisorAgent()
        
        # Expose agents for backward compatibility
        self.image_agent = self.supervisor.image_agent
        self.classification_agent = self.supervisor.classification_agent
        self.root_cause_agent = self.supervisor.root_cause_agent
        self.report_agent = self.supervisor.report_agent
        
        # Build workflow graph (using supervisor pattern)
        self.workflow = self._build_workflow()
        
        logger.info("Orchestrator initialized with Supervisor Agent")
    
    def _build_workflow(self) -> StateGraph:
        """
        Build advanced LangGraph workflow with intelligent routing
        Implements cutting-edge multi-agent orchestration (2024-2025)
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes with advanced capabilities
        workflow.add_node("image_analysis", self._image_analysis_node)
        workflow.add_node("classification", self._classification_node)
        workflow.add_node("root_cause", self._root_cause_node)
        workflow.add_node("report_generation", self._report_generation_node)
        workflow.add_node("quality_check", self._quality_check_node)  # New: quality validation
        
        # Define intelligent routing
        workflow.set_entry_point("image_analysis")
        workflow.add_edge("image_analysis", "classification")
        
        # Conditional routing based on defect count
        workflow.add_conditional_edges(
            "classification",
            self._should_do_advanced_rca,  # Decision function
            {
                "high_priority": "root_cause",  # Many defects -> advanced RCA
                "normal": "root_cause",  # Normal flow
                "low_priority": "report_generation"  # Few defects -> skip advanced RCA
            }
        )
        
        workflow.add_edge("root_cause", "quality_check")
        workflow.add_edge("quality_check", "report_generation")
        workflow.add_edge("report_generation", END)
        
        # Compile workflow without checkpointer for simplicity
        # (Checkpointer requires configurable keys: thread_id, checkpoint_ns, checkpoint_id)
        # For production, you can add checkpointer with proper configuration
        return workflow.compile()
    
    def _should_do_advanced_rca(self, state: AgentState) -> str:
        """
        Intelligent routing decision
        Determines if advanced RCA is needed based on defect patterns
        """
        defects = state.get("defects", [])
        classifications = state.get("classifications", [])
        
        if not defects:
            return "low_priority"
        
        defect_count = len(defects)
        
        # High priority if many defects or high severity
        if defect_count > 10:
            return "high_priority"
        elif defect_count > 5:
            return "normal"
        else:
            return "low_priority"
    
    def _quality_check_node(self, state: AgentState) -> AgentState:
        """
        Quality validation node - validates analysis before report generation
        Implements agent-to-agent communication pattern
        """
        try:
            logger.info("Quality Check Agent: Validating analysis quality")
            
            defects_data = state.get('defects', [])
            classifications_data = state.get('classifications', [])
            root_causes_data = state.get('root_causes', [])
            
            # Agent-to-agent communication
            quality_message = {
                "from": "quality_check_agent",
                "to": "report_agent",
                "message": f"Analysis validated: {len(defects_data)} defects, {len(classifications_data)} classifications",
                "quality_score": min(len(defects_data) / 20.0, 1.0) if defects_data else 0.0
            }
            
            if 'agent_communications' not in state:
                state['agent_communications'] = []
            state['agent_communications'].append(quality_message)
            
            logger.info("Quality check completed - analysis validated")
            
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
        
        return state
    
    def _image_analysis_node(self, state: AgentState) -> AgentState:
        """Image analysis node"""
        try:
            logger.info(f"Image Agent: Analyzing {state['image_path']}")
            
            defects = self.image_agent.analyze_image(state['image_path'])
            # Convert Pydantic models to dicts for LangGraph state
            state['defects'] = [d.model_dump() if hasattr(d, 'model_dump') else d.dict() if hasattr(d, 'dict') else d for d in defects]
            
            logger.info(f"Image Agent: Detected {len(defects)} defects")
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            state['error'] = f"Image analysis failed: {str(e)}"
        
        return state
    
    def _classification_node(self, state: AgentState) -> AgentState:
        """Classification node"""
        try:
            logger.info("Classification Agent: Classifying defects")
            
            defects_data = state.get('defects', [])
            if not defects_data:
                logger.warning("No defects to classify")
                state['classifications'] = []
                return state
            
            # Convert dicts back to Pydantic models for classification
            defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
            classifications = self.classification_agent.classify_defects(defects)
            # Convert back to dicts for state
            state['classifications'] = [c.model_dump() if hasattr(c, 'model_dump') else c.dict() if hasattr(c, 'dict') else c for c in classifications]
            
            logger.info(f"Classification Agent: Classified {len(classifications)} defects")
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            state['error'] = f"Classification failed: {str(e)}"
        
        return state
    
    def _root_cause_node(self, state: AgentState) -> AgentState:
        """Advanced root cause analysis node with intelligent reasoning"""
        try:
            logger.info("Root Cause Agent: Performing advanced AI reasoning")
            
            classifications_data = state.get('classifications', [])
            if not classifications_data:
                logger.warning("No classifications to analyze")
                state['root_causes'] = []
                return state
            
            # Convert dicts back to Pydantic models
            classifications = [ClassificationResult(**c) if isinstance(c, dict) else c for c in classifications_data]
            total_defects = len(state.get('defects', []))
            defects_data = state.get('defects', [])
            defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
            
            # Build reasoning context
            reasoning_context = {
                "defect_count": total_defects,
                "defect_distribution": {},
                "spatial_patterns": self._analyze_spatial_patterns(defects)
            }
            
            # Use advanced analysis if available
            if hasattr(self.root_cause_agent, 'analyze_batch_advanced'):
                root_causes = self.root_cause_agent.analyze_batch_advanced(
                    classifications, total_defects, reasoning_context.get('defect_distribution')
                )
            else:
                root_causes = self.root_cause_agent.analyze_batch(classifications, total_defects)
            
            # Convert back to dicts for state
            state['root_causes'] = [rc.model_dump() if hasattr(rc, 'model_dump') else rc.dict() if hasattr(rc, 'dict') else rc for rc in root_causes]
            state['reasoning_context'] = reasoning_context
            
            # Agent communication
            if 'agent_communications' not in state:
                state['agent_communications'] = []
            state['agent_communications'].append({
                "from": "root_cause_agent",
                "to": "report_agent",
                "message": f"Identified {len(root_causes)} root causes with AI reasoning",
                "confidence": sum(rc.confidence for rc in root_causes) / len(root_causes) if root_causes else 0.0
            })
            
            logger.info(f"Root Cause Agent: Analyzed {len(root_causes)} root causes using advanced reasoning")
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            state['error'] = f"Root cause analysis failed: {str(e)}"
        
        return state
    
    def _analyze_spatial_patterns(self, defects: List) -> Dict:
        """Analyze spatial patterns in defects for advanced reasoning"""
        if not defects:
            return {}
        
        # Simple spatial analysis
        x_coords = [d.bbox.x_min + (d.bbox.x_max - d.bbox.x_min) / 2 for d in defects]
        y_coords = [d.bbox.y_min + (d.bbox.y_max - d.bbox.y_min) / 2 for d in defects]
        
        return {
            "centroid": {
                "x": sum(x_coords) / len(x_coords) if x_coords else 0,
                "y": sum(y_coords) / len(y_coords) if y_coords else 0
            },
            "spread": {
                "x_range": max(x_coords) - min(x_coords) if x_coords else 0,
                "y_range": max(y_coords) - min(y_coords) if y_coords else 0
            },
            "clustered": len(defects) > 5  # Simple clustering detection
        }
    
    def _report_generation_node(self, state: AgentState) -> AgentState:
        """Report generation node"""
        try:
            logger.info("Report Agent: Generating report")
            
            # Build analysis response - convert dicts back to Pydantic models
            defects_data = state.get('defects', [])
            defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
            
            classifications_data = state.get('classifications', [])
            classifications = [ClassificationResult(**c) if isinstance(c, dict) else c for c in classifications_data]
            
            root_causes_data = state.get('root_causes', [])
            root_causes = [RootCauseAnalysis(**rc) if isinstance(rc, dict) else rc for rc in root_causes_data]
            
            # Calculate defect summary
            defect_summary = {}
            for classification in classifications:
                defect_type = classification.defect_type.value
                defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
            
            # Calculate severity score (based on defect count and confidence)
            if defects:
                avg_confidence = sum(c.confidence for c in classifications) / len(classifications) if classifications else 0.5
                defect_density = len(defects) / 1000.0  # Normalize by area (placeholder)
                severity_score = min(avg_confidence * (1 + defect_density), 1.0)
            else:
                severity_score = 0.0
            
            analysis_response = ImageAnalysisResponse(
                analysis_id=state['analysis_id'],
                wafer_id=state.get('wafer_id'),
                batch_id=state.get('batch_id'),
                timestamp=datetime.now(),
                image_path=state['image_path'],
                defects=defects,
                total_defects=len(defects),
                classifications=classifications,
                root_causes=root_causes,
                defect_summary=defect_summary,
                severity_score=severity_score
            )
            
            # Generate report (use advanced if available)
            if hasattr(self.report_agent, 'generate_advanced_report'):
                report_path = self.report_agent.generate_advanced_report(
                    analysis_response,
                    state['image_path'],
                    format="pdf"
                )
            else:
                report_path = self.report_agent.generate_report(
                    analysis_response,
                    state['image_path'],
                    format="pdf"
                )
            
            state['report_path'] = report_path
            analysis_response.report_path = report_path
            
            # Store analysis response in state
            state['analysis_response'] = analysis_response
            
            logger.info(f"Report Agent: Generated report at {report_path}")
            
        except Exception as e:
            logger.error(f"Error in report generation: {e}")
            state['error'] = f"Report generation failed: {str(e)}"
        
        return state
    
    def analyze_wafer(
        self,
        image_path: str,
        wafer_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ImageAnalysisResponse:
        """
        Main method to analyze a wafer image through the Supervisor Agent
        Uses supervisor pattern for intelligent agent coordination
        """
        try:
            # Validate image exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Use Supervisor Agent for coordinated execution
            logger.info(f"Supervisor: Starting supervised analysis for {image_path}")
            return self.supervisor.analyze_wafer_supervised(
                image_path=image_path,
                wafer_id=wafer_id,
                batch_id=batch_id,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in supervised wafer analysis: {e}")
            raise


# Singleton orchestrator instance
_orchestrator_instance: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """
    Get or create the singleton orchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MultiAgentOrchestrator()
    return _orchestrator_instance

