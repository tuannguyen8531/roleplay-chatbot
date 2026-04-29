"""
Diary Node — Extracts facts from conversation for long-term memory.

Runs periodically (every N turns) to analyze recent conversation
and extract important facts about the user for cross-session memory.
"""

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.config import config
from app.models.state import RoleplayState
from app.memory.facts_store import save_facts
from app.logger import log_ai_call

# Module-level mongo_client — set by main.py at startup
_mongo_client = None


def set_mongo_client(client):
    """Set the MongoDB client for facts saving. Called once at startup."""
    global _mongo_client
    _mongo_client = client


def _get_extraction_llm() -> ChatOllama:
    """Get a low-temperature LLM for fact extraction."""
    return ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=0.2,  # Very low for factual extraction
        num_ctx=config.ollama_num_ctx,
    )


def should_run_diary(state: RoleplayState) -> str:
    """
    Conditional edge: decide whether to run diary or skip to END.

    Runs every DIARY_INTERVAL turns.
    """
    turn_count = state.get("turn_count", 0)
    if turn_count > 0 and turn_count % config.diary_interval == 0:
        return "diary"
    return "end"


def diary_node(state: RoleplayState) -> dict:
    """
    Extract facts from recent conversation and save to MongoDB.

    Reads recent messages and uses LLM to identify important facts
    about the user (preferences, decisions, relationships, etc.)
    Passes existing facts to LLM to avoid duplicates.
    """
    if _mongo_client is None:
        return {}

    messages = state["messages"]
    if len(messages) < 4:  # Need at least 2 exchanges
        return {}

    llm = _get_extraction_llm()

    # Load existing facts so LLM knows what's already recorded
    existing_facts = state.get("long_term_facts", [])
    existing_facts_text = ""
    if existing_facts:
        existing_facts_text = "\nAlready known facts (DO NOT repeat these):\n"
        existing_facts_text += "\n".join(f"- {f}" for f in existing_facts)
        existing_facts_text += "\n"

    # Format recent messages
    recent = messages[-10:]  # Last 5 exchanges
    conversation_text = ""
    for msg in recent:
        role = "User" if msg.type == "human" else state["character_name"]
        conversation_text += f"{role}: {msg.content}\n"

    # Extract facts
    prompt = f"""Analyze this roleplay conversation and extract NEW important facts about the User (NOT the character).

{conversation_text}
{existing_facts_text}
Extract facts in these categories:
- User preferences (what they like/dislike in the story)
- Important decisions the user made
- Things the user revealed about themselves
- Key plot events that happened

Rules:
- Only extract CLEAR facts, not assumptions
- Each fact should be one short sentence
- Do NOT repeat any already known facts listed above
- If no NEW facts, respond with exactly "NONE"
- Maximum 5 facts

Respond with only the facts, one per line, no numbering or bullets."""

    log_ai_call(
        "diary",
        character=state["character_name"],
        messages_analyzed=len(recent),
        existing_facts=existing_facts,
        prompt=prompt,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if content.upper() == "NONE" or not content:
        log_ai_call("diary", facts=[])
        return {}

    # Parse facts — filter out NONE lines and empty lines
    new_facts = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and line.strip().upper() != "NONE"
    ]

    if not new_facts:
        log_ai_call("diary", facts=[])
        return {}

    log_ai_call(
        "diary",
        facts=new_facts,
    )

    # Save to MongoDB
    save_facts(
        _mongo_client,
        facts=new_facts,
        character_name=state["character_name"],
    )

    return {}
