import json
import uuid
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.agent.tools import AGENT_TOOLS
from src.config import LLM_MODEL, OLLAMA_BASE_URL

# 1. Initialize the LLM and bind our specific tools to it
# Temperature is set to 0 to prevent the LLM from hallucinating financial figures
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
llm_with_tools = llm.bind_tools(AGENT_TOOLS)

# Define the System Prompt so the agent knows its persona and boundaries
SYSTEM_PROMPT = """You are a highly professional, AI-powered Loan Advisory Agent for a Bank.
Your goal is to assist users with loan-related queries (policies, EMIs, eligibility, fees) using ONLY the provided tools.

CRITICAL RULES:
1. ALWAYS use the 'search_loan_policies' tool first if the user asks about interest rates, fees, eligibility, or bank rules. DO NOT guess.
2. ALWAYS use the 'calculate_emi' tool if the user asks for a monthly payment or EMI calculation.
3. INFER FROM CONTEXT (CRITICAL): If a user asks about a specific profile (e.g., "student", "freelancer", "intern") and the exact word is not in the documents, DO NOT say you don't know. Instead, provide the general eligibility criteria (e.g., minimum age, income requirements, employment continuity) and explain how they apply to the user's situation.
4. If the documents contain absolutely no related information to the overall topic, then state: "I apologize, but I do not have that specific information in my current policy documents."
5. Be polite, concise, and use markdown formatting (bullet points, bold text) to structure your answers beautifully.
"""

# 2. Define the Agent Node
def chatbot(state: AgentState):
    """
    This is the core reasoning node. It takes the conversation history,
    prepends the system instructions, and asks the LLM what to do next.
    """
    messages = state["messages"]
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # We pass the system message + the entire conversation history to the LLM
    response = llm_with_tools.invoke([sys_msg] + messages)
    
    # --- PRODUCTION GUARDRAIL: Local LLM JSON Fallback ---
    # Catch instances where the local LLM outputs raw JSON instead of native tool calls
    if not response.tool_calls and isinstance(response.content, str) and response.content.strip().startswith("{"):
        try:
            parsed = json.loads(response.content.strip())
            if "name" in parsed and "arguments" in parsed:
                # Inject the parsed JSON into LangChain's official tool_calls attribute
                response.tool_calls = [{
                    "name": parsed["name"],
                    "args": parsed["arguments"],
                    "id": "fallback_call"
                }]
                response.content = ""  # Hide the raw JSON so it doesn't clutter the UI
        except json.JSONDecodeError:
            pass  # If it's not valid JSON, let it pass through normally
    # -----------------------------------------------------
    
    return {"messages": [response]}

# 3. Build the LangGraph State Machine
workflow = StateGraph(AgentState)

# Add the nodes (the "actors" in our graph)
workflow.add_node("agent", chatbot)
workflow.add_node("tools", ToolNode(AGENT_TOOLS))

# Set the starting point
workflow.set_entry_point("agent")

# Add conditional routing:
# After the agent thinks, 'tools_condition' checks if the LLM decided to use a tool.
# If yes -> route to 'tools' node. If no -> route to END (it finished answering).
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# If it went to the 'tools' node, it MUST go back to the 'agent' node 
# so the LLM can read the tool's output and format a natural response.
workflow.add_edge("tools", "agent")

# Compile the final graph into an executable application
app = workflow.compile()