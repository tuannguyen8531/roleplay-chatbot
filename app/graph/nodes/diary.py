"""
Diary Node — Extracts facts from conversation for long-term memory.

Runs periodically (every N turns) to analyze recent conversation
and extract important facts about the user for cross-session memory.
"""

from langchain_core.messages import HumanMessage

from app.config import config
from app.models.state import RoleplayState
from app.memory.facts_store import load_facts, save_facts
from app.logger import log_ai_call
from app.llm import get_llm

# Module-level mongo_client — set by main.py at startup
_mongo_client = None


def set_mongo_client(client):
    """Set the MongoDB client for facts saving. Called once at startup."""
    global _mongo_client
    _mongo_client = client


def _get_extraction_llm():
    """Get a low-temperature LLM for fact extraction.
    Uses utility_model if configured for faster processing."""
    return get_llm(
        model_name=config.utility_model,
        temperature=0.2,  # Very low for factual extraction
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

    # Reload freshest facts from DB to avoid duplicates from concurrent/background runs
    existing_facts = load_facts(
        _mongo_client,
        character_name=state["character_name"],
    )
    existing_facts_text = ""
    if existing_facts:
        existing_facts_text = "\nAlready known facts (DO NOT repeat or rephrase these):\n"
        existing_facts_text += "\n".join(f"- {f}" for f in existing_facts)
        existing_facts_text += "\n"

    # Format recent messages
    recent = messages[-10:]  # Last 5 exchanges
    character_name = state["character_name"]
    conversation_text = ""
    for msg in recent:
        role = "User" if msg.type == "human" else character_name
        conversation_text += f"{role}: {msg.content}\n"

    # Extract facts with improved prompt + few-shot examples
    prompt = f"""Extract NEW important facts from this roleplay conversation. Focus on the User (the human player), not {character_name}.

Conversation:
{conversation_text}
{existing_facts_text}
Good facts (specific, actionable):
- User's character name is "Zen von Pendragon"
- User chose to enter the cave instead of the forest
- User prefers diplomatic solutions over combat
- User revealed they are searching for a lost artifact

Bad facts (vague, negative, or obvious — do NOT write these):
- "User did not reveal anything" (negative/empty)
- "User is chatting with {character_name}" (obvious)
- "User responded to a question" (trivial)

Rules:
- Only write CONCRETE, SPECIFIC facts
- Each fact = one short sentence
- Skip if nothing meaningful happened
- Maximum 5 facts
- If no new facts, respond with exactly "NONE"

Respond with only the facts, one per line, no numbering or bullets."""

    log_ai_call(
        "diary",
        character=character_name,
        messages_analyzed=len(recent),
        existing_facts=existing_facts,
        prompt=prompt,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if content.upper() == "NONE" or not content:
        log_ai_call("diary", facts=[])
        return {}

    # Parse facts — filter out NONE lines, empty lines, and negative/vague facts
    new_facts = []
    negative_phrases = [
        "did not reveal", "did not state", "did not mention",
        "no information", "not reveal", "not explicitly",
        "did not express", "no clear",
    ]
    for line in content.split("\n"):
        fact = line.strip().lstrip("•-· ").strip()
        if not fact or fact.upper() == "NONE":
            continue
        # Filter out negative/vague facts
        if any(phrase in fact.lower() for phrase in negative_phrases):
            continue
        new_facts.append(fact)

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
        character_name=character_name,
    )

    return {}
