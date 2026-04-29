"""
Generate Response Node — Calls Ollama LLM to generate the character's response.

This is the core node that:
1. Builds the full prompt (system + conversation history)
2. Calls Ollama via langchain-ollama
3. Returns the AI response to be appended to messages
"""

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from app.config import config
from app.models.state import RoleplayState
from app.models.character import get_character

# Initialize LLMs (cached by temperature)
_llms: dict[float, ChatOllama] = {}


def _get_llm(temperature: float | None = None) -> ChatOllama:
    """Get or create the ChatOllama instance for a specific temperature."""
    t = temperature if temperature is not None else config.ollama_temperature
    if t not in _llms:
        _llms[t] = ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=t,
            num_ctx=config.ollama_num_ctx,
        )
    return _llms[t]


def generate_node(state: RoleplayState) -> dict:
    """
    Generate a response from the character using Ollama.

    Reads:
    - state["character_prompt"] → system message
    - state["messages"] → conversation history

    Writes:
    - Appends AI response to messages
    """
    # Look up character to get specific temperature
    character = get_character(state["character_name"])
    temp = character.temperature if character else None

    llm = _get_llm(temperature=temp)

    # Build the system message from character prompt
    system_msg = SystemMessage(content=state["character_prompt"])

    # Combine: [system_prompt, ...conversation_history]
    all_messages = [system_msg] + state["messages"]

    # Call Ollama
    response = llm.invoke(all_messages)

    return {"messages": [response]}
