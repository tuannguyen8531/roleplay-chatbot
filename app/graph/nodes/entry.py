"""
Entry Node — Prepares context before calling the LLM.

Responsibilities:
1. Load long-term facts from MongoDB
2. Search conversation archive (Qdrant) for relevant past context
3. Summarize old messages when conversation exceeds threshold
4. Trim message history to stay within context window
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama

from app.config import config
from app.models.state import RoleplayState
from app.memory.facts_store import load_facts
from app.memory.conversation_archive import search_archive, get_embedding
from app.logger import log_ai_call

# Module-level clients — set by main.py at startup
_mongo_client = None
_qdrant_client = None


def set_mongo_client(client):
    """Set the MongoDB client for facts loading. Called once at startup."""
    global _mongo_client
    _mongo_client = client


def set_qdrant_client(client):
    """Set the Qdrant client for conversation archive search. Called once at startup."""
    global _qdrant_client
    _qdrant_client = client


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
        _recall_embeddings = [get_embedding(ex) for ex in _recall_examples]
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


def _get_summary_llm() -> ChatOllama:
    """Get a low-temperature LLM for summarization (factual, not creative).
    Uses utility_model if configured for faster processing."""
    return ChatOllama(
        model=config.utility_model,
        base_url=config.ollama_base_url,
        temperature=0.3,  # Low temp for accurate summarization
        num_ctx=config.ollama_num_ctx,
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

    log_ai_call(
        "summarize",
        existing_summary=existing_summary,
        messages_to_summarize=len(messages),
        prompt=prompt,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    summary = response.content.strip()

    log_ai_call(
        "summarize",
        summary=summary,
    )

    return summary


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

        # Use embedding classifier to detect recall intent (semantic, not keyword-based)
        _load_recall_embeddings()
        if latest_user_msg and _needs_rag_by_embedding(latest_user_msg):
            results = search_archive(
                _qdrant_client,
                query=latest_user_msg,
                character_name=state["character_name"],
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
                    results_count=len(results),
                    top_score=results[0]["score"],
                )

    updates["retrieved_context"] = retrieved_context

    # 3. Summarize + trim if over threshold
    if len(messages) > max_messages:
        # Keep half the threshold — creates buffer before next trigger
        keep_count = max_messages // 2

        # Messages to summarize (the old ones we're about to remove)
        messages_to_remove = messages[:-keep_count]

        # Generate summary from old messages
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
