"""
LangGraph-based Supervisor Agent
Implements proper LangGraph supervisor pattern for multi-agent coordination
Based on: https://docs.langchain.com/oss/python/langgraph/workflows-agents
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from pathlib import Path

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app.agents.image_agent import ImageAgent
from app.agents.classification_agent import ClassificationAgent
from app.agents.mapping_agent import MappingAgent
from app.agents.root_cause_agent import RootCauseAgent
from app.agents.report_agent import ReportAgent

# Advanced agents (optional)
try:
    from app.agents.advanced_root_cause_agent import AdvancedRootCauseAgent
    from app.agents.advanced_report_agent import AdvancedReportAgent
    ADVANCED_AGENTS_AVAILABLE = True
except (ImportError, OSError, Exception):
    ADVANCED_AGENTS_AVAILABLE = False
    AdvancedRootCauseAgent = None
    AdvancedReportAgent = None

from app.models.schemas import (
    DefectDetection,
    ClassificationResult,
    RootCauseAnalysis,
    ImageAnalysisResponse
)

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict):
    """
    State schema for LangGraph supervisor workflow
    Follows LangGraph best practices
    """
    image_path: str
    wafer_id: Optional[str]
    batch_id: Optional[str]
    defects: List[Dict]  # Store as dicts for LangGraph compatibility
    classifications: List[Dict]
    defect_map: Optional[Dict]
    root_causes: List[Dict]
    report_path: Optional[str]
    analysis_id: str
    metadata: Dict
    error: Optional[str]
    next_agent: Optional[str]  # For supervisor routing
    agent_results: Dict[str, Any]  # Track results from each agent


class LangGraphSupervisorAgent:
    """
    LangGraph-based Supervisor Agent
    Implements proper LangGraph supervisor pattern for multi-agent coordination
    Reference: https://docs.langchain.com/oss/python/langgraph/workflows-agents
    """
    
    def __init__(self):
        """Initialize Supervisor Agent with all sub-agents"""
        logger.info("Initializing LangGraph Supervisor Agent...")
        
        # Initialize all agents
        self.image_agent = ImageAgent()
        self.classification_agent = ClassificationAgent()
        self.mapping_agent = MappingAgent()
        
        # Use advanced agents if available
        if ADVANCED_AGENTS_AVAILABLE:
            try:
                self.root_cause_agent = AdvancedRootCauseAgent()
                logger.info("✓ Using Advanced Root Cause Agent")
            except Exception as e:
                logger.warning(f"Advanced RCA agent failed, using standard: {e}")
                self.root_cause_agent = RootCauseAgent()
            
            try:
                self.report_agent = AdvancedReportAgent()
                logger.info("✓ Using Advanced Report Agent")
            except Exception as e:
                logger.warning(f"Advanced report agent failed, using standard: {e}")
                self.report_agent = ReportAgent()
        else:
            self.root_cause_agent = RootCauseAgent()
            self.report_agent = ReportAgent()
        
        # Build LangGraph workflow
        self.workflow = self._build_supervisor_workflow()
        
        logger.info("LangGraph Supervisor Agent initialized successfully")
    
    def _build_supervisor_workflow(self) -> StateGraph:
        """
        Build LangGraph supervisor workflow
        Implements proper supervisor pattern with agent nodes
        """
        workflow = StateGraph(SupervisorState)
        
        # Add agent nodes
        workflow.add_node("image_agent", self._image_agent_node)
        workflow.add_node("classification_agent", self._classification_agent_node)
        workflow.add_node("mapping_agent", self._mapping_agent_node)
        workflow.add_node("root_cause_agent", self._root_cause_agent_node)
        workflow.add_node("report_agent", self._report_agent_node)
        workflow.add_node("supervisor", self._supervisor_node)  # Supervisor routing node
        
        # Define workflow edges
        workflow.set_entry_point("supervisor")
        
        # Supervisor routes to first agent
        workflow.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            {
                "image_agent": "image_agent",
                "classification_agent": "classification_agent",
                "mapping_agent": "mapping_agent",
                "root_cause_agent": "root_cause_agent",
                "report_agent": "report_agent",
                "end": END
            }
        )
        
        # Agent execution flow
        workflow.add_edge("image_agent", "supervisor")
        workflow.add_edge("classification_agent", "supervisor")
        workflow.add_edge("mapping_agent", "supervisor")
        workflow.add_edge("root_cause_agent", "supervisor")
        workflow.add_edge("report_agent", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """
        Supervisor node - decides which agent to execute next
        Implements intelligent routing logic
        """
        logger.info("Supervisor: Routing to next agent...")
        
        # Determine next agent based on current state
        if not state.get("defects"):
            # No defects yet - route to image agent
            state["next_agent"] = "image_agent"
        elif not state.get("classifications"):
            # Defects found but not classified - route to classification agent
            state["next_agent"] = "classification_agent"
        elif not state.get("defect_map"):
            # Classified but not mapped - route to mapping agent
            state["next_agent"] = "mapping_agent"
        elif not state.get("root_causes"):
            # Classified but no root causes - route to root cause agent
            state["next_agent"] = "root_cause_agent"
        elif not state.get("report_path"):
            # All analysis done - route to report agent
            state["next_agent"] = "report_agent"
        else:
            # All done
            state["next_agent"] = "end"
        
        logger.info(f"Supervisor: Routing to {state['next_agent']}")
        return state
    
    def _route_to_agent(self, state: SupervisorState) -> str:
        """Route to the next agent based on supervisor decision"""
        return state.get("next_agent", "end")
    
    def _image_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Image Agent node - detects defects"""
        try:
            logger.info("Supervisor: Executing Image Agent...")
            defects = self.image_agent.analyze_image(state["image_path"])
            
            # Convert Pydantic models to dicts
            state["defects"] = [
                d.model_dump() if hasattr(d, 'model_dump') else d.dict() if hasattr(d, 'dict') else d
                for d in defects
            ]
            
            state["agent_results"]["image_agent"] = {
                "status": "completed",
                "defects_count": len(defects)
            }
            
            logger.info(f"Supervisor: Image Agent detected {len(defects)} defects")
            
        except Exception as e:
            logger.error(f"Image Agent failed: {e}")
            state["error"] = f"Image analysis failed: {str(e)}"
            state["agent_results"]["image_agent"] = {"status": "failed", "error": str(e)}
        
        return state
    
    def _classification_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Classification Agent node - classifies defects"""
        try:
            logger.info("Supervisor: Executing Classification Agent...")
            
            # Convert dicts back to Pydantic models
            defects_data = state.get("defects", [])
            defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
            
            classifications = self.classification_agent.classify_defects(defects)
            
            # Convert back to dicts
            state["classifications"] = [
                c.model_dump() if hasattr(c, 'model_dump') else c.dict() if hasattr(c, 'dict') else c
                for c in classifications
            ]
            
            state["agent_results"]["classification_agent"] = {
                "status": "completed",
                "classifications_count": len(classifications)
            }
            
            logger.info(f"Supervisor: Classification Agent classified {len(classifications)} defects")
            
        except Exception as e:
            logger.error(f"Classification Agent failed: {e}")
            state["error"] = f"Classification failed: {str(e)}"
            state["agent_results"]["classification_agent"] = {"status": "failed", "error": str(e)}
        
        return state
    
    def _mapping_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Mapping Agent node - creates defect map"""
        try:
            logger.info("Supervisor: Executing Mapping Agent...")
            
            # Convert dicts back to Pydantic models
            defects_data = state.get("defects", [])
            defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
            
            classifications_data = state.get("classifications", [])
            classifications = [ClassificationResult(**c) if isinstance(c, dict) else c for c in classifications_data]
            
            # Create defect map
            defect_map_result = self.mapping_agent.create_defect_map(
                defects=defects,
                classifications=classifications,
                image_path=state["image_path"],
                analysis_id=state["analysis_id"]
            )
            
            state["defect_map"] = defect_map_result
            state["agent_results"]["mapping_agent"] = {
                "status": "completed",
                "clusters_count": len(defect_map_result.get("clusters", []))
            }
            
            logger.info(f"Supervisor: Mapping Agent created map with {len(defect_map_result.get('clusters', []))} clusters")
            
        except Exception as e:
            logger.error(f"Mapping Agent failed: {e}")
            state["error"] = f"Mapping failed: {str(e)}"
            state["defect_map"] = None
            state["agent_results"]["mapping_agent"] = {"status": "failed", "error": str(e)}
        
        return state
    
    def _root_cause_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Root Cause Agent node - analyzes root causes"""
        try:
            logger.info("Supervisor: Executing Root Cause Agent...")
            
            # Convert dicts back to Pydantic models
            classifications_data = state.get("classifications", [])
            classifications = [ClassificationResult(**c) if isinstance(c, dict) else c for c in classifications_data]
            
            total_defects = len(state.get("defects", []))
            
            if hasattr(self.root_cause_agent, 'analyze_batch_advanced'):
                root_causes = self.root_cause_agent.analyze_batch_advanced(
                    classifications, total_defects, {}
                )
            else:
                root_causes = self.root_cause_agent.analyze_batch(classifications, total_defects)
            
            # Convert back to dicts
            state["root_causes"] = [
                rc.model_dump() if hasattr(rc, 'model_dump') else rc.dict() if hasattr(rc, 'dict') else rc
                for rc in root_causes
            ]
            
            state["agent_results"]["root_cause_agent"] = {
                "status": "completed",
                "root_causes_count": len(root_causes)
            }
            
            logger.info(f"Supervisor: Root Cause Agent analyzed {len(root_causes)} root causes")
            
        except Exception as e:
            logger.error(f"Root Cause Agent failed: {e}")
            state["error"] = f"Root cause analysis failed: {str(e)}"
            state["agent_results"]["root_cause_agent"] = {"status": "failed", "error": str(e)}
        
        return state
    
    def _report_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Report Agent node - generates report"""
        try:
            logger.info("Supervisor: Executing Report Agent...")
            
            # Build analysis response
            defects_data = state.get("defects", [])
            defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
            
            classifications_data = state.get("classifications", [])
            classifications = [ClassificationResult(**c) if isinstance(c, dict) else c for c in classifications_data]
            
            root_causes_data = state.get("root_causes", [])
            root_causes = [RootCauseAnalysis(**rc) if isinstance(rc, dict) else rc for rc in root_causes_data]
            
            # Calculate summary
            defect_summary = {}
            for classification in classifications:
                if hasattr(classification, 'defect_type'):
                    defect_type = classification.defect_type.value
                    defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
            
            avg_confidence = sum(c.confidence for c in classifications) / len(classifications) if classifications else 0.5
            defect_density = len(defects) / 1000.0
            severity_score = min(avg_confidence * (1 + defect_density), 1.0)
            
            from app.models.schemas import DefectMapData
            
            defect_map_data = state.get("defect_map")
            
            analysis_response = ImageAnalysisResponse(
                analysis_id=state["analysis_id"],
                wafer_id=state.get("wafer_id"),
                batch_id=state.get("batch_id"),
                timestamp=datetime.now(),
                image_path=state["image_path"],
                defects=defects,
                total_defects=len(defects),
                classifications=classifications,
                root_causes=root_causes,
                defect_summary=defect_summary,
                severity_score=severity_score,
                defect_map=DefectMapData(**defect_map_data) if defect_map_data else None
            )
            
            # Generate report
            if hasattr(self.report_agent, 'generate_advanced_report'):
                report_path = self.report_agent.generate_advanced_report(
                    analysis_response, state["image_path"], "pdf"
                )
            else:
                report_path = self.report_agent.generate_report(
                    analysis_response, state["image_path"], "pdf"
                )
            
            state["report_path"] = report_path
            state["agent_results"]["report_agent"] = {
                "status": "completed",
                "report_path": report_path
            }
            
            logger.info(f"Supervisor: Report Agent generated report at {report_path}")
            
        except Exception as e:
            logger.error(f"Report Agent failed: {e}")
            state["error"] = f"Report generation failed: {str(e)}"
            state["agent_results"]["report_agent"] = {"status": "failed", "error": str(e)}
        
        return state
    
    def analyze_wafer_supervised(
        self,
        image_path: str,
        wafer_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ImageAnalysisResponse:
        """
        Main supervised analysis workflow using LangGraph
        Coordinates all agents through LangGraph supervisor pattern
        """
        analysis_id = str(uuid.uuid4())
        logger.info(f"Supervisor: Starting LangGraph supervised analysis {analysis_id}")
        
        # Initialize state
        initial_state: SupervisorState = {
            "image_path": image_path,
            "wafer_id": wafer_id,
            "batch_id": batch_id,
            "defects": [],
            "classifications": [],
            "defect_map": None,
            "root_causes": [],
            "report_path": None,
            "analysis_id": analysis_id,
            "metadata": metadata or {},
            "error": None,
            "next_agent": "image_agent",
            "agent_results": {}
        }
        
        # Execute LangGraph workflow
        final_state = self.workflow.invoke(initial_state, config={})
        
        # Check for errors
        if final_state.get("error"):
            raise Exception(final_state["error"])
        
        # Build final response
        defects_data = final_state.get("defects", [])
        defects = [DefectDetection(**d) if isinstance(d, dict) else d for d in defects_data]
        
        classifications_data = final_state.get("classifications", [])
        classifications = [ClassificationResult(**c) if isinstance(c, dict) else c for c in classifications_data]
        
        root_causes_data = final_state.get("root_causes", [])
        root_causes = [RootCauseAnalysis(**rc) if isinstance(rc, dict) else rc for rc in root_causes_data]
        
        defect_summary = {}
        for classification in classifications:
            if hasattr(classification, 'defect_type'):
                defect_type = classification.defect_type.value
                defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
        
        avg_confidence = sum(c.confidence for c in classifications) / len(classifications) if classifications else 0.5
        defect_density = len(defects) / 1000.0
        severity_score = min(avg_confidence * (1 + defect_density), 1.0)
        
        from app.models.schemas import DefectMapData
        
        defect_map_data = final_state.get("defect_map")
        
        response = ImageAnalysisResponse(
            analysis_id=analysis_id,
            wafer_id=wafer_id,
            batch_id=batch_id,
            timestamp=datetime.now(),
            image_path=image_path,
            defects=defects,
            total_defects=len(defects),
            classifications=classifications,
            root_causes=root_causes,
            defect_summary=defect_summary,
            severity_score=severity_score,
            report_path=final_state.get("report_path"),
            defect_map=DefectMapData(**defect_map_data) if defect_map_data else None
        )
        
        logger.info(f"Supervisor: LangGraph analysis {analysis_id} completed")
        logger.info(f"Supervisor: Agent results: {final_state.get('agent_results', {})}")
        
        return response
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "supervisor": "active",
            "agents": {
                "image_agent": "ready",
                "classification_agent": "ready",
                "mapping_agent": "ready",
                "root_cause_agent": "ready",
                "report_agent": "ready"
            },
            "workflow": "langgraph_supervisor"
        }

