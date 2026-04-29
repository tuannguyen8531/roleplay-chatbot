"""
Generate Response Node — Calls Ollama LLM to generate the character's response.

This is the core node that:
1. Builds the full prompt (system + summary + facts + conversation history)
2. Calls Ollama via langchain-ollama
3. Returns the AI response to be appended to messages
"""

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from app.config import config
from app.models.state import RoleplayState
from app.models.character import get_character
from app.logger import log_ai_call

import threading

# Thread-local cache: each thread gets its own ChatOllama instance
_llm_local = threading.local()


def _get_llm(temperature: float | None = None) -> ChatOllama:
    """Get or create the ChatOllama instance for a specific temperature."""
    t = temperature if temperature is not None else config.ollama_temperature
    if not hasattr(_llm_local, "llms"):
        _llm_local.llms = {}
    if t not in _llm_local.llms:
        _llm_local.llms[t] = ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=t,
            num_ctx=config.ollama_num_ctx,
        )
    return _llm_local.llms[t]


def _build_system_prompt(state: RoleplayState) -> str:
    """Build the full system prompt with character, summary, and facts."""
    base_prompt = state["character_prompt"]

    # Append conversation summary if available
    summary = state.get("conversation_summary", "")
    if summary:
        base_prompt += f"\n\n## Previous Conversation Summary\n{summary}"

    # Append long-term facts if available
    facts = state.get("long_term_facts", [])
    if facts:
        facts_str = "\n".join(f"- {f}" for f in facts)
        base_prompt += f"\n\n## Known Facts About This User\n{facts_str}"

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

    llm = _get_llm(temperature=temp)

    # Build enriched system prompt (character + summary + facts)
    system_prompt = _build_system_prompt(state)
    system_msg = SystemMessage(content=system_prompt)

    # Combine: [system_prompt, ...conversation_history]
    all_messages = [system_msg] + state["messages"]

    # Log the input
    log_ai_call(
        "generate",
        character=state["character_name"],
        system_prompt=system_prompt,
        messages_count=len(state["messages"]),
        messages=[{"role": m.type, "content": m.content} for m in state["messages"]],
    )

    # Call Ollama
    response = llm.invoke(all_messages)

    # Log the response
    log_ai_call(
        "generate",
        character=state["character_name"],
        response=response.content,
    )

    return {"messages": [response]}
