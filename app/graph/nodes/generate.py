"""
Generate Response Node — Calls Ollama LLM to generate the character's response.

This is the core node that:
1. Builds the full prompt (system + summary + facts + conversation history)
2. Calls Ollama via langchain-ollama
3. Returns the AI response to be appended to messages
"""

from langchain_core.messages import SystemMessage
from app.config import config
from app.models.state import RoleplayState
from app.models.character import get_character
from app.logger import log_ai_request, log_ai_response
from app.llm import get_llm


def _build_system_prompt(state: RoleplayState) -> str:
    """Build the full system prompt with character, summary, facts, and RAG context."""
    base_prompt = state["character_prompt"]

    # Append conversation summary if available
    summary = state.get("conversation_summary", "")
    if summary:
        base_prompt += f"\n\n## Previous Conversation Summary\n{summary}"

    # Append long-term facts if available (limit to last 15 facts to avoid context bloat)
    facts = state.get("long_term_facts", [])
    if facts:
        recent_facts = facts[-15:]
        facts_str = "\n".join(f"- {f}" for f in recent_facts)
        base_prompt += f"\n\n## Known Facts About This User\n{facts_str}"

    # Append retrieved past conversation context (from RAG)
    retrieved = state.get("retrieved_context", "")
    if retrieved:
        base_prompt += (
            f"\n\n## Relevant Past Conversations\n"
            f"The following are past exchanges that may be relevant to the current conversation. "
            f"Use them to maintain consistency and recall details accurately.\n\n{retrieved}"
        )

    return base_prompt


def generate_node(state: RoleplayState) -> dict:
    """
    Generate a response from the character using Ollama.

    Reads:
    - state["character_prompt"] → base system message
    - state["conversation_summary"] → summary of older messages
    - state["long_term_facts"] → cross-session facts
    - state["messages"] → recent conversation history

    Writes:
    - Appends AI response to messages
    """
    # Look up character to get specific temperature
    character = get_character(state["character_name"])
    temp = character.temperature if character else None

    llm = get_llm(temperature=temp)

    # Build enriched system prompt (character + summary + facts)
    system_prompt = _build_system_prompt(state)
    system_msg = SystemMessage(content=system_prompt)

    # Combine: [system_prompt, ...conversation_history]
    all_messages = [system_msg] + state["messages"]

    # Log the input
    call_id = log_ai_request(
        "generate",
        character=state["character_name"],
        system_prompt=system_prompt,
        messages_count=len(state["messages"]),
        messages=[{"role": m.type, "content": m.content} for m in state["messages"]],
    )

    # Call Ollama
    response = llm.invoke(all_messages)

    # Log the response
    log_ai_response(
        "generate",
        call_id=call_id,
        character=state["character_name"],
        response=response.content,
    )

    return {"messages": [response]}
