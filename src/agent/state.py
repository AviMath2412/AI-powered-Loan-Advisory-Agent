from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state/memory of our LangGraph agent at any given moment.
    
    The 'messages' list stores the entire conversation.
    'Annotated' and 'add_messages' tell LangGraph to append new messages 
    to this list rather than overwriting the whole list on every turn.
    """
    messages: Annotated[list[BaseMessage], add_messages]