import os
import json
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from src.memory import get_checkpointer
from src.llm_factory import get_llm
from src.agent.resilience import invoke_llm_with_resilience
from src.agent.pipeline_state import (
    PipelineState, IntentData, ConstraintData, MemoryData, DocumentData,
    ConflictData, MissingInfoData, ConfidenceData, PolicyValidationData,
    ReasoningData, GroundingData, FormattedResponseData
)
from src.observability import trace_node

def _last_human_message(state: PipelineState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.type == "human":
            return msg.content
    return ""

def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    return get_llm(provider=provider, model=model, temperature=0)

@trace_node("intent_detection")
def intent_detection(state: PipelineState):
    msg = _last_human_message(state)
    return {"intent": IntentData(needs_research="loan" in msg.lower(), needs_calculation="calculate" in msg.lower())}

@trace_node("constraint_extraction")
def constraint_extraction(state: PipelineState):
    return {"constraints": ConstraintData(user_constraints=[])}

@trace_node("memory_retrieval")
def memory_retrieval(state: PipelineState):
    return {"memory": MemoryData(profile={"name": "User"}, past_history=[])}

@trace_node("document_retrieval")
def document_retrieval(state: PipelineState):
    return {"documents": DocumentData(retrieved_docs="Policy Docs")}

@trace_node("conflict_detection")
def conflict_detection(state: PipelineState):
    return {"conflicts": ConflictData(conflicts_found=[])}

@trace_node("missing_information_detection")
def missing_information_detection(state: PipelineState):
    return {"missing_info": MissingInfoData(missing_fields=[])}

@trace_node("confidence_estimation")
def confidence_estimation(state: PipelineState):
    return {"confidence": ConfidenceData(score=0.9, reasoning="Good info")}

@trace_node("policy_validation")
def policy_validation(state: PipelineState):
    return {"policy_validation": PolicyValidationData(hard_requirements_met=True, violations=[])}

@trace_node("reasoning_agent")
def reasoning_agent(state: PipelineState):
    llm = _get_llm()
    msg = _last_human_message(state)
    prompt = f"Answer the user query concisely: {msg}"
    response = invoke_llm_with_resilience(llm, [SystemMessage(content=prompt)], fallback_response="Fallback")
    return {"reasoning": ReasoningData(draft=response)}

@trace_node("grounding_verification")
def grounding_verification(state: PipelineState):
    draft = state.get("reasoning", ReasoningData()).draft
    return {"grounding": GroundingData(verified_draft=draft)}

@trace_node("response_formatter")
def response_formatter(state: PipelineState):
    verified = state.get("grounding", GroundingData()).verified_draft
    return {"formatted_response": FormattedResponseData(final_output=f"**Agent Response:**\n{verified}")}

@trace_node("final_response")
def final_response(state: PipelineState):
    final = state.get("formatted_response", FormattedResponseData()).final_output
    return {"messages": [AIMessage(content=final)]}

# Build Pipeline
workflow = StateGraph(PipelineState)
workflow.add_node("intent_detection", intent_detection)
workflow.add_node("constraint_extraction", constraint_extraction)
workflow.add_node("memory_retrieval", memory_retrieval)
workflow.add_node("document_retrieval", document_retrieval)
workflow.add_node("conflict_detection", conflict_detection)
workflow.add_node("missing_information_detection", missing_information_detection)
workflow.add_node("confidence_estimation", confidence_estimation)
workflow.add_node("policy_validation", policy_validation)
workflow.add_node("reasoning_agent", reasoning_agent)
workflow.add_node("grounding_verification", grounding_verification)
workflow.add_node("response_formatter", response_formatter)
workflow.add_node("final_response", final_response)

workflow.set_entry_point("intent_detection")
workflow.add_edge("intent_detection", "constraint_extraction")
workflow.add_edge("constraint_extraction", "memory_retrieval")
workflow.add_edge("memory_retrieval", "document_retrieval")
workflow.add_edge("document_retrieval", "conflict_detection")
workflow.add_edge("conflict_detection", "missing_information_detection")
workflow.add_edge("missing_information_detection", "confidence_estimation")
workflow.add_edge("confidence_estimation", "policy_validation")
workflow.add_edge("policy_validation", "reasoning_agent")
workflow.add_edge("reasoning_agent", "grounding_verification")
workflow.add_edge("grounding_verification", "response_formatter")
workflow.add_edge("response_formatter", "final_response")
workflow.add_edge("final_response", END)

app = workflow.compile(checkpointer=get_checkpointer())
