import uuid
import pytest
from langchain_core.messages import HumanMessage
from src.agent.pipeline_graph import app
from src.agent.pipeline_state import PipelineState, IntentData

def test_pipeline_refactor_execution():
    """Verify that the refactored 13-stage pipeline executes cleanly."""
    state = {
        "messages": [HumanMessage(content="I want a loan of $5000")],
        "intent": IntentData(),
    }
    
    # Execute the graph
    thread_id = f"test_pipeline_{uuid.uuid4().hex}"
    result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    
    # Assert nodes ran and populated the strictly typed Pydantic state fields
    assert "messages" in result
    assert len(result["messages"]) == 2
    assert "intent" in result
    assert result["intent"].needs_research == True  # "loan" in prompt
    assert result["confidence"].score == 0.9
    assert result["policy_validation"].hard_requirements_met == True
    assert "**Agent Response:**" in result["formatted_response"].final_output
