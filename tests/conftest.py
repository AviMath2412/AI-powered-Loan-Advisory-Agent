import re
import json
import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _planner_json(query="loan", needs_calc=False):
    return json.dumps({
        "needs_research": True,
        "needs_calculation": needs_calc,
        "needs_credit_check": False,
        "search_query": query,
        "calc_params": None,
        "applicant_id": None,
        "profile_updates": {},
        "new_constraints": []
    })


def _extract_user_question(prompt_text: str) -> str:
    """Pull the 'User question:' line out of the synthesizer prompt."""
    match = re.search(r"user question:\s*(.+?)(?:\n|$)", prompt_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return ""


def _synthesizer_response(system_content: str) -> str:
    """Route synthesizer calls based on the embedded user question."""
    prompt_lower = system_content.lower()
    q = _extract_user_question(system_content)

    # --- contradictory profile ---
    if "6200" in q or ("qualify" in q and "contradict" in prompt_lower):
        return "There appears to be a contradiction in the income information. Please clarify the conflict."

    # --- missing info (refinance) ---
    if "refinanc" in q:
        return "Missing information: the remaining balance is required to evaluate refinancing."

    # --- impossible loan (age 17) ---
    if "can i get a loan" in q and ("17" in prompt_lower or "age" in prompt_lower):
        return "You cannot get a loan. The minimum age hard requirement is 18 years."

    # --- partial retrieval: tenure + interest rate (must be before plain interest rate check) ---
    if "tenure" in q and "interest rate" in q:
        return "The interest rate is 5%. However, information about maximum tenure is missing and not provided."

    # --- conflicting policies (interest rate only) ---
    if "interest rate" in q:
        return "There is a conflict between retrieved documents: Doc1 states 5% and Doc2 states 7%."

    # --- prompt injection ---
    if "ignore all previous" in q or ("approve" in q and "10m" in q):
        return "I cannot override my instructions or approve unauthorized loans."

    # --- hallucination attempt (spaceship) ---
    if "spaceship" in q or ("crypto" in q and "backed" in q):
        return "I do not have information about this. It is not mentioned in our policy documents."

    # --- multiple currencies: 'Do I pass?' (check BEFORE EUR mismatch to avoid collision) ---
    if "do i pass" in q:
        return "Cannot evaluate. There is a currency mismatch between EUR and GBP policy requirements."

    # --- currency mismatch EUR (4500 EUR specific amount) ---
    if "eur" in q and "4500" in q:
        return "There is a currency mismatch. Income is in EUR but the policy minimum is in USD."

    # --- zero income ---
    if "income" in q and "eligible" in q:
        return "You are not eligible. Your income of 0 does not meet the minimum (must be > $0)."

    # --- negative savings ---
    if "advance" in q:
        return "You are not eligible. Applicants must have positive savings. Negative savings disqualify the applicant."

    # --- conflicting dates ---
    if "offer" in q and ("2024-05-01" in q or "today is" in q):
        return "The offer expired on 2024-01-01. You cannot use this offer."

    # --- empty retrieval ---
    if "policy on x" in q:
        return "No relevant policy documents found. Cannot provide information on Policy X."

    # (partial retrieval check moved above interest rate check)

    # --- retrieval timeout ---
    if "tell me about loans" in q:
        return "The research service is currently unavailable due to a connection timeout error."

    # --- duplicate retrieval ---
    if "rules" in q:
        return "The applicable rule is: Rule A."

    # --- tool failure (EMI with broken calc) ---
    if "emi" in q or ("calculate" in q and "10000" in q):
        return "Cannot complete the calculation. The calculation tool is unavailable or returned an error."

    return "Based on available information, I can provide the following guidance for your query."


# ---------------------------------------------------------------------------
# Main stub — dispatches to planner (JSON) vs synthesizer (prose) by message shape
# ---------------------------------------------------------------------------
def _stub_llm_response(llm=None, messages=None, circuit_breaker=None, max_retries=None,
                       initial_delay=None, backoff_factor=None, fallback_response=None,
                       **kwargs) -> str:
    if messages is None:
        messages = []

    # Planner call: messages list contains a HumanMessage
    for m in messages:
        if isinstance(m, HumanMessage):
            user_text = (m.content or "").lower()
            needs_calc = any(w in user_text for w in ["calculate", "emi", "10000", "amortiz"])
            return _planner_json(query=user_text[:80].replace('"', "'"), needs_calc=needs_calc)

    # Synthesizer / other node: only SystemMessage(s) present
    system_content = ""
    for m in messages:
        content = getattr(m, "content", "") or ""
        if content:
            system_content = content
            break

    return _synthesizer_response(system_content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def stub_invoke_llm():
    """
    Patch invoke_llm_with_resilience at its IMPORT SITE in graph.py.
    Patching the resilience module itself does NOT work because graph.py
    holds its own reference via 'from src.agent.resilience import ...'.
    """
    with patch("src.agent.graph.invoke_llm_with_resilience", side_effect=_stub_llm_response):
        yield


@pytest.fixture(autouse=True)
def disable_circuit_breaker():
    """Ensure the circuit breaker never blocks calls during tests."""
    with patch("src.agent.resilience.CircuitBreaker.can_execute", return_value=True):
        yield
