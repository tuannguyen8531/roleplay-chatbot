"""
Entry Node — The first node in the graph.

Phase 1: Simply passes through (character prompt is set at invocation time).
Phase 2: Trims message history to stay within context window limits.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import config
from app.models.state import RoleplayState


def entry_node(state: RoleplayState) -> dict:
    """
    Prepare context before calling the LLM.

    Phase 2: Trims messages to keep conversation within context window.
    Keeps the most recent N messages (configured via max_history_messages).
    """
    messages = state["messages"]
    max_messages = config.max_history_messages

    # If within limits, nothing to do
    if len(messages) <= max_messages:
        return {}

    # Trim: keep only the last N messages
    trimmed = messages[-max_messages:]

    return {"messages": trimmed}
