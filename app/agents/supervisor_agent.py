"""
Supervisor Agent - Coordinates all wafer defect analysis agents
Implements intelligent routing, result aggregation, and agent coordination
"""
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

from app.agents.image_agent import ImageAgent
from app.agents.classification_agent import ClassificationAgent
from app.agents.root_cause_agent import RootCauseAgent
from app.agents.report_agent import ReportAgent
from app.agents.mapping_agent import MappingAgent

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


class AgentStatus(str, Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentResult:
    """Result from an agent execution"""
    def __init__(
        self,
        agent_name: str,
        status: AgentStatus,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0.0
    ):
        self.agent_name = agent_name
        self.status = status
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.timestamp = datetime.now()


class SupervisorAgent:
    """
    Supervisor Agent - Coordinates all wafer defect analysis agents
    Implements intelligent routing, result aggregation, and agent coordination
    """
    
    def __init__(self):
        """Initialize Supervisor Agent with all sub-agents"""
        logger.info("Initializing Supervisor Agent...")
        
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
        
        # Agent registry
        self.agents = {
            "image_agent": self.image_agent,
            "classification_agent": self.classification_agent,
            "mapping_agent": self.mapping_agent,
            "root_cause_agent": self.root_cause_agent,
            "report_agent": self.report_agent
        }
        
        # Execution tracking
        self.execution_history: List[AgentResult] = []
        self.current_execution: Dict[str, AgentResult] = {}
        
        logger.info("Supervisor Agent initialized successfully")
    
    def route_analysis(self, image_path: str, wafer_id: Optional[str] = None, 
                      batch_id: Optional[str] = None) -> List[str]:
        """
        Intelligent routing - determines which agents to execute and in what order
        Returns list of agent names in execution order
        """
        # Standard workflow: Image -> Classification -> Mapping -> Root Cause -> Report
        route = ["image_agent", "classification_agent", "mapping_agent", "root_cause_agent", "report_agent"]
        
        # Future: Could add conditional routing based on:
        # - Image type (SEM vs optical)
        # - Defect count threshold
        # - User preferences
        # - Historical patterns
        
        return route
    
    def execute_agent(self, agent_name: str, **kwargs) -> AgentResult:
        """
        Execute a single agent with error handling and timing
        """
        import time
        start_time = time.time()
        
        agent = self.agents.get(agent_name)
        if not agent:
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error=f"Agent {agent_name} not found"
            )
        
        try:
            logger.info(f"Supervisor: Executing {agent_name}...")
            self.current_execution[agent_name] = AgentResult(
                agent_name=agent_name,
                status=AgentStatus.RUNNING
            )
            
            # Route to appropriate agent method
            if agent_name == "image_agent":
                result = agent.analyze_image(kwargs.get('image_path'))
            elif agent_name == "classification_agent":
                result = agent.classify_defects(kwargs.get('defects', []))
            elif agent_name == "mapping_agent":
                result = agent.create_defect_map(
                    defects=kwargs.get('defects', []),
                    classifications=kwargs.get('classifications', []),
                    image_path=kwargs.get('image_path'),
                    analysis_id=kwargs.get('analysis_id', '')
                )
            elif agent_name == "root_cause_agent":
                classifications = kwargs.get('classifications', [])
                total_defects = kwargs.get('total_defects', 0)
                if hasattr(agent, 'analyze_batch_advanced'):
                    result = agent.analyze_batch_advanced(
                        classifications, total_defects, kwargs.get('defect_distribution', {})
                    )
                else:
                    result = agent.analyze_batch(classifications, total_defects)
            elif agent_name == "report_agent":
                analysis_response = kwargs.get('analysis_response')
                image_path = kwargs.get('image_path')
                if hasattr(agent, 'generate_advanced_report'):
                    result = agent.generate_advanced_report(analysis_response, image_path, "pdf")
                else:
                    result = agent.generate_report(analysis_response, image_path, "pdf")
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            execution_time = time.time() - start_time
            
            agent_result = AgentResult(
                agent_name=agent_name,
                status=AgentStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )
            
            self.current_execution[agent_name] = agent_result
            self.execution_history.append(agent_result)
            
            logger.info(f"Supervisor: {agent_name} completed in {execution_time:.2f}s")
            return agent_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{agent_name} failed: {str(e)}"
            logger.error(error_msg)
            
            agent_result = AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
            
            self.current_execution[agent_name] = agent_result
            self.execution_history.append(agent_result)
            
            return agent_result
    
    def aggregate_results(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """
        Aggregate results from all agents into a unified response
        """
        aggregated = {
            "defects": [],
            "classifications": [],
            "defect_map": None,
            "root_causes": [],
            "report_path": None,
            "execution_summary": {}
        }
        
        # Extract results from each agent
        if "image_agent" in results and results["image_agent"].status == AgentStatus.COMPLETED:
            aggregated["defects"] = results["image_agent"].result or []
        
        if "classification_agent" in results and results["classification_agent"].status == AgentStatus.COMPLETED:
            aggregated["classifications"] = results["classification_agent"].result or []
        
        if "mapping_agent" in results and results["mapping_agent"].status == AgentStatus.COMPLETED:
            aggregated["defect_map"] = results["mapping_agent"].result
        
        if "root_cause_agent" in results and results["root_cause_agent"].status == AgentStatus.COMPLETED:
            aggregated["root_causes"] = results["root_cause_agent"].result or []
        
        if "report_agent" in results and results["report_agent"].status == AgentStatus.COMPLETED:
            aggregated["report_path"] = results["report_agent"].result
        
        # Execution summary
        for agent_name, result in results.items():
            aggregated["execution_summary"][agent_name] = {
                "status": result.status.value,
                "execution_time": result.execution_time,
                "error": result.error
            }
        
        return aggregated
    
    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of agents (all or specific)
        """
        if agent_name:
            if agent_name in self.current_execution:
                result = self.current_execution[agent_name]
                return {
                    "agent": agent_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error": result.error,
                    "timestamp": result.timestamp.isoformat()
                }
            return {"agent": agent_name, "status": "not_found"}
        
        # Return all agents status
        status = {}
        for name in self.agents.keys():
            if name in self.current_execution:
                result = self.current_execution[name]
                status[name] = {
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "error": result.error
                }
            else:
                status[name] = {"status": "pending"}
        
        return status
    
    def analyze_wafer_supervised(
        self,
        image_path: str,
        wafer_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ImageAnalysisResponse:
        """
        Main supervised analysis workflow
        Coordinates all agents through the supervisor pattern
        """
        from datetime import datetime
        import uuid
        
        analysis_id = str(uuid.uuid4())
        logger.info(f"Supervisor: Starting supervised analysis {analysis_id}")
        
        # Clear previous execution
        self.current_execution = {}
        
        # Route analysis
        agent_route = self.route_analysis(image_path, wafer_id, batch_id)
        logger.info(f"Supervisor: Execution route: {' -> '.join(agent_route)}")
        
        # Execute agents in sequence
        results: Dict[str, AgentResult] = {}
        
        # 1. Image Agent
        image_result = self.execute_agent("image_agent", image_path=image_path)
        results["image_agent"] = image_result
        
        if image_result.status != AgentStatus.COMPLETED:
            raise Exception(f"Image analysis failed: {image_result.error}")
        
        defects = image_result.result
        if not defects:
            logger.warning("Supervisor: No defects detected, skipping further analysis")
            # Still generate a report with no defects
        
        # 2. Classification Agent
        classification_result = self.execute_agent(
            "classification_agent",
            defects=defects
        )
        results["classification_agent"] = classification_result
        
        if classification_result.status != AgentStatus.COMPLETED:
            logger.warning(f"Classification failed: {classification_result.error}")
            classifications = []
        else:
            classifications = classification_result.result
        
        # 3. Mapping Agent
        mapping_result = self.execute_agent(
            "mapping_agent",
            defects=defects,
            classifications=classifications,
            image_path=image_path,
            analysis_id=analysis_id
        )
        results["mapping_agent"] = mapping_result
        
        if mapping_result.status != AgentStatus.COMPLETED:
            logger.warning(f"Mapping failed: {mapping_result.error}")
            defect_map_data = None
        else:
            defect_map_data = mapping_result.result
        
        # 4. Root Cause Agent
        root_cause_result = self.execute_agent(
            "root_cause_agent",
            classifications=classifications,
            total_defects=len(defects),
            defect_distribution={}
        )
        results["root_cause_agent"] = root_cause_result
        
        if root_cause_result.status != AgentStatus.COMPLETED:
            logger.warning(f"Root cause analysis failed: {root_cause_result.error}")
            root_causes = []
        else:
            root_causes = root_cause_result.result
        
        # 5. Build analysis response
        defect_summary = {}
        for classification in classifications:
            if hasattr(classification, 'defect_type'):
                defect_type = classification.defect_type.value
                defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
        
        avg_confidence = sum(c.confidence for c in classifications) / len(classifications) if classifications else 0.5
        defect_density = len(defects) / 1000.0
        severity_score = min(avg_confidence * (1 + defect_density), 1.0)
        
        analysis_response = ImageAnalysisResponse(
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
            severity_score=severity_score
        )
        
        # Add defect map to response
        if defect_map_data:
            from app.models.schemas import DefectMapData
            analysis_response.defect_map = DefectMapData(**defect_map_data)
        
        # 6. Report Agent
        report_result = self.execute_agent(
            "report_agent",
            analysis_response=analysis_response,
            image_path=image_path
        )
        results["report_agent"] = report_result
        
        if report_result.status == AgentStatus.COMPLETED:
            analysis_response.report_path = report_result.result
        
        # Aggregate results
        aggregated = self.aggregate_results(results)
        logger.info(f"Supervisor: Analysis {analysis_id} completed")
        logger.info(f"Supervisor: Execution summary: {aggregated['execution_summary']}")
        
        return analysis_response

