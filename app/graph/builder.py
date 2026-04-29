"""
Graph Builder — Assembles the LangGraph state machine.

Current flow:
    START → entry (load facts, summarize, trim)
          → generate (call Ollama)
          → END

Diary runs asynchronously in a background thread after generate,
so the user doesn't have to wait for fact extraction.
"""

import threading

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.entry import entry_node
from app.graph.nodes.generate import generate_node
from app.graph.nodes.diary import diary_node, should_run_diary
from app.models.state import RoleplayState


def _post_generate_node(state: RoleplayState) -> dict:
    """
    Wrapper that runs after generate.
    Launches diary in background thread if conditions are met.
    """
    if should_run_diary(state) == "diary":
        # Run diary in background — doesn't block user response
        thread = threading.Thread(
            target=diary_node,
            args=(state,),
            daemon=True,
        )
        thread.start()

    return {}


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the roleplay chatbot graph.

    Flow:
        START → entry → generate → post_generate (async diary) → END

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
    """
    graph = StateGraph(RoleplayState)

    # Add nodes
    graph.add_node("entry", entry_node)
    graph.add_node("generate", generate_node)
    graph.add_node("post_generate", _post_generate_node)

    # Wire edges — linear flow, diary runs async inside post_generate
    graph.add_edge(START, "entry")
    graph.add_edge("entry", "generate")
    graph.add_edge("generate", "post_generate")
    graph.add_edge("post_generate", END)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)
