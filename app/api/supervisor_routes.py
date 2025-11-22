"""
Additional routes for Supervisor Agent monitoring and control
"""
from fastapi import APIRouter
from typing import Optional
from app.core.orchestrator import get_orchestrator

router = APIRouter(prefix="/supervisor", tags=["supervisor"])


@router.get("/status")
async def get_supervisor_status(agent_name: Optional[str] = None):
    """
    Get Supervisor Agent status and all sub-agent statuses
    """
    orch = get_orchestrator()
    status = orch.supervisor.get_agent_status()
    return {
        "supervisor": "active",
        "workflow_type": "langgraph_supervisor",
        "agents": status.get("agents", {}),
        "total_agents": len(status.get("agents", {}))
    }


@router.get("/history")
async def get_execution_history(limit: int = 10):
    """
    Get recent agent execution history
    Note: LangGraph supervisor tracks execution in state, not separate history
    """
    return {
        "message": "Execution history tracked in workflow state",
        "workflow": "langgraph_supervisor",
        "note": "Check agent_results in analysis response for execution details"
    }

