from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, validator
import json

class CalcParams(BaseModel):
    principal: float = Field(..., gt=0, le=100000000, description="Loan principal amount")
    rate_pa: float = Field(..., gt=0, le=50, description="Annual interest rate percentage")
    tenure_months: int = Field(..., gt=0, le=360, description="Loan tenure in months")

class PlannerOutput(BaseModel):
    needs_research: bool = Field(default=False)
    needs_calculation: bool = Field(default=False)
    needs_credit_check: bool = Field(default=False)
    search_query: Optional[str] = None
    calc_params: Optional[CalcParams] = None
    applicant_id: Optional[str] = None
    profile_updates: Dict[str, Any] = Field(default_factory=dict)
    new_constraints: List[str] = Field(default_factory=list, description="Any new explicit constraints the user requested (e.g. 'Do not suggest X')")

class CriticOutput(BaseModel):
    is_adequate: bool = Field(..., description="Whether the evidence is adequate to answer the user query")
    feedback: str = Field(..., description="Feedback for the researcher if inadequate")

class PolicyStatement(BaseModel):
    statement: str = Field(..., description="The policy statement extracted from text")
    type: Literal["hard_requirement", "recommendation", "preference"] = Field(..., description="The classification of the policy")

class ClassifiedEvidence(BaseModel):
    policies: List[PolicyStatement] = Field(default_factory=list, description="List of classified policy statements")

class ValidationOutput(BaseModel):
    conflicts: List[str] = Field(default_factory=list, description="Conflicting user information or contradictory documents")
    missing_information: List[str] = Field(default_factory=list, description="Missing information required to proceed")
    unsafe_assumptions: List[str] = Field(default_factory=list, description="Unsupported assumptions made")
    constraint_violations: List[str] = Field(default_factory=list, description="Impossible requests or mutually exclusive constraints")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    confidence_reasoning: List[str] = Field(default_factory=list, description="Reasons for the confidence score (e.g. number of retrieved docs, conflicts, etc.)")
    can_answer: bool = Field(..., description="Whether the agent can answer the user's request")

class ConstraintCheckOutput(BaseModel):
    violated: bool = Field(..., description="Whether any user constraints were violated by the draft")
    feedback: str = Field(..., description="Explanation of which constraints were violated and how to fix them")

def validate_llm_json(raw_json_str: str, model_class: type[BaseModel], default_instance: BaseModel) -> BaseModel:
    """
    Attempts to parse and validate a raw JSON string into a Pydantic model.
    Falls back to a default instance if validation or parsing fails.
    """
    try:
        # Strip potential markdown formatting
        cleaned = raw_json_str.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return model_class(**data)
    except Exception as e:
        # Log failure here if needed
        return default_instance
