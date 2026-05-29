"""
Entry Node — Prepares context before calling the LLM.

Responsibilities:
1. Load long-term facts from MongoDB
2. Search conversation archive (Qdrant) for relevant past context
3. Summarize old messages when conversation exceeds threshold
4. Trim message history to stay within context window
"""

import logging
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

from app.config import config
from app.models.state import RoleplayState
from app.memory.facts_store import load_facts
from app.memory.conversation_archive import search_archive, get_embedding, index_batch, get_embeddings
from app.logger import log_ai_call, log_ai_request, log_ai_response
from app.llm import get_llm

# Module-level clients — set by main.py at startup
_mongo_client = None
_qdrant_client = None
_bg_executor = None
_archive_logger = logging.getLogger("archive")


def set_mongo_client(client):
    """Set the MongoDB client for facts loading. Called once at startup."""
    global _mongo_client
    _mongo_client = client


def set_qdrant_client(client):
    """Set the Qdrant client for conversation archive search. Called once at startup."""
    global _qdrant_client
    _qdrant_client = client


def set_bg_executor(executor):
    """Set the background thread executor. Called once at startup."""
    global _bg_executor
    _bg_executor = executor


def _safe_index_batch(client, messages, character_name, thread_id):
    """Safely index a batch of conversation messages in a background task."""
    try:
        index_batch(client, messages, character_name, thread_id)
    except Exception:
        _archive_logger.exception("Unexpected background archive indexing failure")


# ── Embedding-based RAG Intent Classifier ────────────────────────────────────

_recall_examples = [
    "What did we talk about before?",
    "Do you remember when I entered the cave?",
    "Who was that blacksmith?",
    "What happened last time?",
    "Tell me about Moonblade again",
    "Did we discuss Vex earlier?",
    "Remind me what I said about the forest",
    "What was their name?",
    "Earlier you mentioned something...",
    "Before, when we were at the tavern...",
    "Refresh my memory about the quest",
    "What did I decide back then?",
    "Can you recall our previous conversation?",
    "What was that thing you told me?",
]

_recall_embeddings: list[list[float]] | None = None


def _load_recall_embeddings():
    """Compute embeddings for recall examples once. Falls back to empty list on failure."""
    global _recall_embeddings
    if _recall_embeddings is not None:
        return
    try:
        _recall_embeddings = get_embeddings(_recall_examples)
    except Exception:
        _recall_embeddings = []  # fallback: classifier disabled


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _needs_rag_by_embedding(query: str) -> bool:
    """Decide if message needs RAG using embedding similarity to recall examples."""
    if not _recall_embeddings:
        return False
    try:
        query_emb = get_embedding(query)
    except Exception:
        return False
    max_sim = max(_cosine_similarity(query_emb, ex) for ex in _recall_embeddings)
    return max_sim > 0.72


_RECALL_KEYWORDS = re.compile(
    r"\b(remember|recall|before|earlier|last time|previous|again|back then|"
    r"did we|did you|did i|what did|who was|what was|tell me about|remind me)\b",
    re.IGNORECASE,
)


def _needs_rag_by_keyword(query: str) -> bool:
    """Fast keyword gate to trigger RAG without embedding cost."""
    return bool(_RECALL_KEYWORDS.search(query))


def _get_summary_llm():
    """Get a low-temperature LLM for summarization (factual, not creative).
    Uses utility_model if configured for faster processing."""
    return get_llm(
        model_name=config.utility_model,
        temperature=0.3,  # Low temp for accurate summarization
    )


def _summarize_messages(messages: list, existing_summary: str, character_name: str) -> str:
    """
    Summarize old messages into a concise paragraph.

    If there's an existing summary, incorporates it into the new summary.
    Includes character name for context-aware summarization.
    """
    llm = _get_summary_llm()

    # Format messages for summarization
    conversation_text = ""
    for msg in messages:
        role = "User" if msg.type == "human" else character_name
        conversation_text += f"{role}: {msg.content}\n"

    # Build summarization prompt
    if existing_summary:
        prompt = f"""Summarize the roleplay conversation between User and {character_name}. Merge the existing summary with new events.

Existing summary:
{existing_summary}

New conversation:
{conversation_text}

Rules:
- Write 3-5 sentences max
- Focus on: plot events, decisions made, relationship changes, locations visited
- Use character name "{character_name}" (not "Character")
- Keep track of user's in-character name if revealed
- No preamble, just the summary"""
    else:
        prompt = f"""Summarize this roleplay conversation between User and {character_name}.

{conversation_text}

Rules:
- Write 3-5 sentences max
- Focus on: plot events, decisions made, relationship changes, locations visited
- Use character name "{character_name}" (not "Character")
- Keep track of user's in-character name if revealed
- No preamble, just the summary"""

    call_id = log_ai_request(
        "summarize",
        existing_summary=existing_summary,
        messages_to_summarize=len(messages),
        prompt=prompt,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    summary = response.content.strip()

    log_ai_response(
        "summarize",
        call_id=call_id,
        summary=summary,
    )

    return summary


def _get_trim_cutoff(messages: list, keep_count: int) -> int:
    """
    Choose a trim boundary that removes only complete human->AI exchanges.

    Entry runs after the latest human message is appended but before the AI
    response exists, so a naive "keep last N" can leave history starting with
    an orphan AI response.
    """
    min_cutoff = max(0, len(messages) - keep_count)
    cutoff = min_cutoff

    while cutoff > 0:
        prev_type = getattr(messages[cutoff - 1], "type", None)
        next_type = (
            getattr(messages[cutoff], "type", None)
            if cutoff < len(messages)
            else None
        )
        if prev_type == "ai" and next_type == "human":
            return cutoff
        cutoff -= 1

    return 0


def entry_node(state: RoleplayState) -> dict:
    """
    Prepare context before calling the LLM.

    1. Loads long-term facts from MongoDB
    2. Searches conversation archive for relevant past context
    3. If messages exceed threshold: summarize old ones, then trim
    4. Increments turn counter
    """
    updates: dict = {}
    messages = state["messages"]
    max_messages = config.max_history_messages
    current_summary = state.get("conversation_summary", "")
    turn_count = state.get("turn_count", 0)
    thread_id = state.get("thread_id", "")

    # 1. Load long-term facts
    if _mongo_client is not None:
        facts = load_facts(
            _mongo_client,
            character_name=state["character_name"],
        )
        updates["long_term_facts"] = facts

    # 2. Search conversation archive for relevant past exchanges
    retrieved_context = ""
    if _qdrant_client is not None and messages:
        # Use the latest user message as query
        latest_user_msg = None
        for msg in reversed(messages):
            if msg.type == "human":
                latest_user_msg = msg.content
                break

        # Hybrid intent detection: keyword gate first, fallback to embedding
        needs_rag = False
        if latest_user_msg:
            needs_rag = _needs_rag_by_keyword(latest_user_msg)
            if not needs_rag:
                _load_recall_embeddings()
                needs_rag = _needs_rag_by_embedding(latest_user_msg)

        if needs_rag:
            results = search_archive(
                _qdrant_client,
                query=latest_user_msg,
                character_name=state["character_name"],
                thread_id=thread_id,
                top_k=3,
                score_threshold=config.rag_score_threshold,
            )
            if results:
                context_parts = []
                for r in results:
                    context_parts.append(
                        f"User: {r['user_message']}\n"
                        f"{state['character_name']}: {r['ai_response']}"
                    )
                retrieved_context = "\n---\n".join(context_parts)

                log_ai_call(
                    "rag_search",
                    query=latest_user_msg,
                    thread_id=thread_id,
                    results_count=len(results),
                    top_score=results[0]["score"],
                )

    updates["retrieved_context"] = retrieved_context

    # 3. Summarize + trim if over threshold
    if len(messages) > max_messages:
        # Keep half the threshold — creates buffer before next trigger
        keep_count = max(1, max_messages // 2)

        # Messages to summarize (the old ones we're about to remove)
        trim_cutoff = _get_trim_cutoff(messages, keep_count)
        messages_to_remove = messages[:trim_cutoff]

        # Index batch into Qdrant before removing (one vector for all exchanges)
        if _qdrant_client is not None and messages_to_remove:
            if _bg_executor is not None:
                _bg_executor.submit(
                    _safe_index_batch,
                    _qdrant_client,
                    messages_to_remove,
                    state["character_name"],
                    thread_id,
                )
            else:
                _safe_index_batch(_qdrant_client, messages_to_remove, state["character_name"], thread_id)

        # Generate summary from old messages
        if messages_to_remove:
            new_summary = _summarize_messages(
                messages_to_remove, current_summary, state["character_name"]
            )
            updates["conversation_summary"] = new_summary

            # Use RemoveMessage to explicitly delete old messages from state
            # (add_messages reducer ignores simple list replacement — must use RemoveMessage)
            updates["messages"] = [RemoveMessage(id=m.id) for m in messages_to_remove]

    # 4. Increment turn count
    updates["turn_count"] = turn_count + 1

    return updates
