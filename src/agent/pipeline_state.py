from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
import operator

# Typed Pydantic models for inter-node communication
class IntentData(BaseModel):
    needs_research: bool = False
    needs_calculation: bool = False

class ConstraintData(BaseModel):
    user_constraints: List[str] = Field(default_factory=list)

class MemoryData(BaseModel):
    profile: dict = Field(default_factory=dict)
    past_history: List[str] = Field(default_factory=list)

class DocumentData(BaseModel):
    retrieved_docs: str = ""

class ConflictData(BaseModel):
    conflicts_found: List[str] = Field(default_factory=list)

class MissingInfoData(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)

class ConfidenceData(BaseModel):
    score: float = 1.0
    reasoning: str = ""

class PolicyValidationData(BaseModel):
    hard_requirements_met: bool = True
    violations: List[str] = Field(default_factory=list)

class ReasoningData(BaseModel):
    draft: str = ""

class GroundingData(BaseModel):
    verified_draft: str = ""

class FormattedResponseData(BaseModel):
    final_output: str = ""

class PipelineState(TypedDict):
    messages: Annotated[list, operator.add]
    intent: IntentData
    constraints: ConstraintData
    memory: MemoryData
    documents: DocumentData
    conflicts: ConflictData
    missing_info: MissingInfoData
    confidence: ConfidenceData
    policy_validation: PolicyValidationData
    reasoning: ReasoningData
    grounding: GroundingData
    formatted_response: FormattedResponseData
