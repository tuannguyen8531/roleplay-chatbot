"""
Graph Builder — Assembles the LangGraph state machine.

Phase 1: Simple linear graph: Entry → Generate → END
Phase 2: + MongoDB checkpointer for persistence + message trimming
Phase 3+: Will add Router, RAG, Tool nodes with conditional edges.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.entry import entry_node
from app.graph.nodes.generate import generate_node
from app.models.state import RoleplayState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the roleplay chatbot graph.

    Phase 2 flow:
        START → entry (trim messages) → generate → END
        + MongoDB checkpointer saves state after each node

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
                      Pass MongoDBSaver to enable cross-session memory.

    Returns a compiled graph that can be invoked with:
        graph.invoke(
            {"messages": [...], "character_name": "...", "character_prompt": "..."},
            config={"configurable": {"thread_id": "some-thread-id"}}
        )
    """
    graph = StateGraph(RoleplayState)

    # Add nodes
    graph.add_node("entry", entry_node)
    graph.add_node("generate", generate_node)

    # Wire edges: START → entry → generate → END
    graph.add_edge(START, "entry")
    graph.add_edge("entry", "generate")
    graph.add_edge("generate", END)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)
