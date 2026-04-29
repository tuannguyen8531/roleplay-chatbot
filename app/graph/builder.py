"""
Graph Builder — Assembles the LangGraph state machine.

Current flow:
    START → entry (load facts, summarize, trim)
          → generate (call Ollama)
          → END

Diary extraction is handled outside the graph by the caller
so it doesn't block the user response.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.entry import entry_node
from app.graph.nodes.generate import generate_node
from app.models.state import RoleplayState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the roleplay chatbot graph.

    Flow:
        START → entry → generate → END

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
    """
    graph = StateGraph(RoleplayState)

    # Add nodes
    graph.add_node("entry", entry_node)
    graph.add_node("generate", generate_node)

    # Wire edges — linear flow
    graph.add_edge(START, "entry")
    graph.add_edge("entry", "generate")
    graph.add_edge("generate", END)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)
