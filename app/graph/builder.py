"""
Graph Builder — Assembles the LangGraph state machine.

Current flow:
    START → entry (load facts, summarize, trim)
          → generate (call Ollama)
          → conditional: diary (every N turns) or END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.entry import entry_node
from app.graph.nodes.generate import generate_node
from app.graph.nodes.diary import diary_node, should_run_diary
from app.models.state import RoleplayState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the roleplay chatbot graph.

    Flow:
        START → entry → generate → [diary every N turns | END]

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
    """
    graph = StateGraph(RoleplayState)

    # Add nodes
    graph.add_node("entry", entry_node)
    graph.add_node("generate", generate_node)
    graph.add_node("diary", diary_node)

    # Wire edges
    graph.add_edge(START, "entry")
    graph.add_edge("entry", "generate")

    # After generate: conditionally run diary or go to END
    graph.add_conditional_edges(
        "generate",
        should_run_diary,
        {"diary": "diary", "end": END},
    )
    graph.add_edge("diary", END)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)
