def get_contradiction_rules() -> str:
    return """
---
CONTRADICTION DETECTION RULES
You must actively detect contradictions across:
1. User profile
2. Retrieved documents (evidence)
3. Current conversation
4. Retrieved memory

Examples of contradictions:
- Income = $6200 vs Profile income = $10000
- Credit score = 590 vs Retrieved document says 710

If contradictions exist:
- Do not silently choose one.
- Explain the contradiction explicitly in your analysis/response.
- Lower your confidence score (if applicable).
- Ask for clarification only when required.
---
"""

def get_missing_info_rules() -> str:
    return """
---
MISSING INFORMATION DETECTOR (CROSS-DOMAIN)
Instead of guessing unknown facts, you must detect which variables are required for the requested reasoning.

Examples of required variables:
- Financial (e.g. Refinancing?): remaining balance, interest rate, tenure, prepayment penalties.
- Medical (e.g. Diagnosis?): symptoms, duration, medical history, current medications.
- Legal (e.g. Breach of contract?): contract terms, dates, evidence of breach, jurisdiction.
- General: any critical dependency needed to form a logical conclusion.

If required information is missing:
1. List the missing fields explicitly.
2. Proceed ONLY if a safe partial answer or conditional assessment is possible.
3. Otherwise, refuse to answer and ask the user to provide the missing fields.
---
"""

def get_numeric_reasoning_rules() -> str:
    return """
---
NUMERIC REASONING GUARDRAILS
You are STRICTLY PROHIBITED from performing mental arithmetic, mathematical calculations, or numeric estimations directly.
1. Never calculate sums, differences, multiplications, divisions, percentages, LTV, APR comparisons, or dates on your own.
2. All numeric outputs MUST be sourced directly from the provided tool outputs (Calculation Result, Credit Result, or explicit Context).
3. If a calculation is requested but the result is not present in the provided context, you MUST state that the calculation requires the backend numeric utility and cannot be performed directly.
---
"""

def append_reasoning_rules(prompt: str) -> str:
    """Appends strict contradiction, missing info, and numeric guardrail rules to any agent prompt."""
    return prompt.strip() + "\n\n" + get_contradiction_rules() + "\n\n" + get_missing_info_rules() + "\n\n" + get_numeric_reasoning_rules()
