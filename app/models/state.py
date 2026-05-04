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
    """

    # Character
    character_name: str
    character_prompt: str

    # Memory
    conversation_summary: str  # Summary of trimmed old messages
    long_term_facts: list[str]  # Facts loaded from DB for this user

    # RAG
    retrieved_context: str  # Relevant past conversation snippets from Qdrant

    # Counters
    turn_count: int  # Track turns for diary trigger
