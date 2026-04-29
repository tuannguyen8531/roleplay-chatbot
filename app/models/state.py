"""
LangGraph State definition for the Roleplay Chatbot.

The State is the central data object that flows through every node in the graph.
Each node reads from and writes to specific fields in this state.
"""

from typing import Annotated

from langgraph.graph import MessagesState


class RoleplayState(MessagesState):
    """
    State schema for the roleplay chatbot graph.

    Extends MessagesState which provides:
    - messages: list of chat messages with auto-append reducer

    Phase 1 fields:
    - character_name: Name of the active character
    - character_prompt: Full system prompt built from character profile

    Future phases will add:
    - conversation_summary: str  (Phase 2 — trimmed history summary)
    - long_term_facts: list[str] (Phase 2 — cross-session facts)
    - intent: str                (Phase 3 — router classification)
    - retrieved_context: str     (Phase 3 — RAG results)
    - tool_results: str          (Phase 3 — tool execution output)
    """

    character_name: str
    character_prompt: str
