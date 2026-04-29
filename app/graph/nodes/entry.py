"""
Entry Node — Prepares context before calling the LLM.

Responsibilities:
1. Load long-term facts from MongoDB
2. Summarize old messages when conversation exceeds threshold
3. Trim message history to stay within context window
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama

from app.config import config
from app.models.state import RoleplayState
from app.memory.facts_store import load_facts
from app.logger import log_ai_call

# Module-level mongo_client — set by main.py at startup
_mongo_client = None


def set_mongo_client(client):
    """Set the MongoDB client for facts loading. Called once at startup."""
    global _mongo_client
    _mongo_client = client


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
    2. If messages exceed threshold: summarize old ones, then trim
    3. Increments turn counter
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

    # 2. Summarize + trim if over threshold
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

    # 3. Increment turn count
    updates["turn_count"] = turn_count + 1

    return updates
