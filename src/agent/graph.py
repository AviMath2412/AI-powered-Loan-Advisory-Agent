import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.tools import search_loan_policies, calculate_emi, check_credit_score
from src.agent.extractors import extract_calc_params, looks_like_calc_request
from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.memory import get_checkpointer

MAX_RETRIES = 2

# Temperature 0 everywhere: every node here does either structured extraction or
# grounded synthesis, never free-form chat, so we want deterministic output throughout.
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _last_human_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return msg.content
    return ""


def _safe_json_parse(raw: str) -> dict:
    """
    Extract a JSON object from an LLM response, tolerating markdown fences or stray
    prose around it. This is the generalized version of the old JSON-fallback guardrail —
    every node here talks to the LLM in structured JSON rather than via native tool-calls,
    so this one parser replaces that single-purpose patch.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are the Planner in a multi-agent loan advisory system.
Read the user's latest message and the known user profile, then output ONLY a JSON object
(no prose, no markdown fences) with this exact shape:

{{
  "needs_research": true/false,
  "needs_calculation": true/false,
  "needs_credit_check": true/false,
  "search_query": "best search query for the policy database, or empty string",
  "calc_params": {{"principal": number, "rate_pa": number, "tenure_months": number}} or null,
  "applicant_id": "string the user gave to identify themselves, or null",
  "profile_updates": {{"age": number or null, "employment_type": "string or null", "monthly_income": number or null, "loan_type_interest": "string or null"}}
}}

Rules:
- needs_research = true whenever the user asks about eligibility, interest rates, fees, documents, or any bank policy.
- needs_calculation = true whenever the user asks for an EMI, monthly payment, or amortization schedule, AND gives (or has previously given) enough numbers to compute it. If numbers are missing, set this false.
- needs_credit_check = true ONLY if the user explicitly asks to check/simulate their credit score. Never infer this on your own.
- Only fill profile_updates fields you can confidently infer from this message; otherwise use null.

Known user profile so far: {profile}
"""


def planner_node(state: AgentState):
    user_msg = _last_human_message(state)
    prompt = PLANNER_PROMPT.format(profile=json.dumps(state.get("user_profile", {})))
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_msg)])
    plan = _safe_json_parse(response.content)

    profile = dict(state.get("user_profile", {}))
    for key, value in (plan.get("profile_updates") or {}).items():
        if value is not None:
            profile[key] = value

    # --- Regex fallback for calc_params -------------------------------------
    # Small local models are unreliable at converting "20 Lakh" -> 2000000 or
    # "5 years" -> 60 inside a JSON blob. If the LLM's extraction is missing or
    # incomplete, and we can parse the numbers ourselves from the raw message,
    # trust the regex extraction instead of leaving the Calculator empty-handed.
    calc_params = plan.get("calc_params")
    calc_params_complete = (
        isinstance(calc_params, dict)
        and all(calc_params.get(k) for k in ("principal", "rate_pa", "tenure_months"))
    )
    regex_params = None
    if not calc_params_complete:
        regex_params = extract_calc_params(user_msg)
        if regex_params:
            calc_params = regex_params

    needs_calculation = bool(plan.get("needs_calculation", False))
    if regex_params and looks_like_calc_request(user_msg):
        needs_calculation = True
    # -------------------------------------------------------------------------

    return {
        "user_profile": profile,
        "needs_research": bool(plan.get("needs_research", False)),
        "needs_calculation": needs_calculation,
        "needs_credit_check": bool(plan.get("needs_credit_check", False)),
        "search_query": plan.get("search_query") or user_msg,
        "calc_params": calc_params,
        "applicant_id": plan.get("applicant_id"),
        "retry_count": 0,
    }


# ---------------------------------------------------------------------------
# Researcher / Calculator / Credit — deterministic execution, no LLM tool-calling.
# Planner already extracted the parameters; these nodes just run the functions.
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState):
    evidence = ""
    if state.get("needs_research"):
        evidence = search_loan_policies.invoke({"query": state["search_query"]})
    
    if state.get("uploaded_doc_text"):
        doc_name = state.get("uploaded_doc_name") or "Uploaded Document"
        evidence += f"\n\n[CONTEXT FROM UPLOADED DOCUMENT ({doc_name}):\n{state['uploaded_doc_text']}\n]"
        
    return {"research_evidence": evidence}


def calculator_node(state: AgentState):
    if not state.get("needs_calculation") or not state.get("calc_params"):
        return {"calculation_result": ""}
    params = state["calc_params"]
    try:
        result = calculate_emi.invoke({
            "principal": float(params.get("principal", 0)),
            "rate_pa": float(params.get("rate_pa", 0)),
            "tenure_months": int(params.get("tenure_months", 0)),
        })
    except (TypeError, ValueError):
        result = "Error: could not parse calculation parameters."
    return {"calculation_result": result}


def credit_node(state: AgentState):
    if not state.get("needs_credit_check"):
        return {"credit_result": ""}
    applicant_id = state.get("applicant_id") or "anonymous"
    result = check_credit_score.invoke({"applicant_id": applicant_id})
    return {"credit_result": result}


# ---------------------------------------------------------------------------
# Critic — self-reflection / evidence sufficiency check
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """You are the Critic in a multi-agent loan advisory system.
Judge whether the evidence below is enough to answer the user's question well.

User question: {question}
Research evidence: {evidence}
Needs research: {needs_research}

Respond ONLY with JSON: {{"verdict": "sufficient" or "retry", "rewritten_query": "a better search query, or empty string"}}

Say "retry" ONLY if research was needed and the evidence is empty or clearly off-topic, AND a
differently worded query is likely to find something better (e.g. "student loan" -> "education financing").
Otherwise say "sufficient" — do not retry just because the evidence is partial.
"""


def critic_node(state: AgentState):
    if not state.get("needs_research"):
        return {"critic_verdict": "sufficient"}

    prompt = CRITIC_PROMPT.format(
        question=_last_human_message(state),
        evidence=(state.get("research_evidence", "") or "")[:2000],
        needs_research=state.get("needs_research"),
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    verdict = _safe_json_parse(response.content)

    if verdict.get("verdict") == "retry" and state.get("retry_count", 0) < MAX_RETRIES:
        return {
            "critic_verdict": "retry",
            "search_query": verdict.get("rewritten_query") or state["search_query"],
            "retry_count": state.get("retry_count", 0) + 1,
        }
    return {"critic_verdict": "sufficient"}


def route_after_critic(state: AgentState):
    return "researcher" if state.get("critic_verdict") == "retry" else "synthesizer"


# ---------------------------------------------------------------------------
# Synthesizer — final, polished answer grounded only in gathered evidence
# ---------------------------------------------------------------------------

SYNTHESIZER_PROMPT = """You are a highly professional AI Loan Advisory Agent for a bank.
Write the final answer to the user using ONLY the evidence and results provided below —
never invent a policy figure, rate, or eligibility rule that isn't in the evidence.
Be polite, concise, and use markdown (bold, bullet points) to structure the answer.
If research was needed but the evidence is empty or irrelevant, say plainly that the
information is not in the current policy documents, rather than guessing.

User question: {question}
User profile: {profile}
Research evidence: {evidence}
Calculation result: {calculation}
Credit check result: {credit}
"""


def synthesizer_node(state: AgentState):
    prompt = SYNTHESIZER_PROMPT.format(
        question=_last_human_message(state),
        profile=json.dumps(state.get("user_profile", {})),
        evidence=state.get("research_evidence") or "None retrieved.",
        calculation=state.get("calculation_result") or "None requested.",
        credit=state.get("credit_result") or "None requested.",
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("calculator", calculator_node)
workflow.add_node("credit", credit_node)
workflow.add_node("critic", critic_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "calculator")
workflow.add_edge("calculator", "credit")
workflow.add_edge("credit", "critic")
workflow.add_conditional_edges("critic", route_after_critic, {
    "researcher": "researcher",
    "synthesizer": "synthesizer",
})
workflow.add_edge("synthesizer", END)

# Compile with a persistent checkpointer: conversations survive across Streamlit reruns
# and process restarts, keyed by thread_id (see src/memory.py).
app = workflow.compile(checkpointer=get_checkpointer())