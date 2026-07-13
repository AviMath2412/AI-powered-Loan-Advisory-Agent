from typing import Annotated, TypedDict, Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserProfile(TypedDict, total=False):
    age: Optional[int]
    employment_type: Optional[str]
    monthly_income: Optional[float]
    loan_type_interest: Optional[str]


class AgentState(TypedDict):
    """
    Shared state for the multi-agent graph.

    'messages' is the conversation history (unchanged from the original single-agent version).
    Everything else is scratch space the Planner/Researcher/Calculator/Critic/Synthesizer
    read from and write to on a single turn. It does not need to persist across turns except
    'user_profile', which the Planner merges into on every message.
    """
    messages: Annotated[list[BaseMessage], add_messages]

    # Persistent across turns (checkpointed)
    user_profile: UserProfile

    # Scratch space, rebuilt each turn by the Planner
    needs_research: bool
    needs_calculation: bool
    needs_credit_check: bool
    search_query: str
    calc_params: Optional[dict]
    applicant_id: Optional[str]

    # Filled in by Researcher / Calculator / Credit nodes
    research_evidence: str
    calculation_result: str
    credit_result: str

    # Critic loop control
    critic_verdict: Literal["sufficient", "retry"]
    retry_count: int