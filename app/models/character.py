"""
Character data model and default character definitions.

Characters are defined as dataclasses for now. Phase 2 will move to YAML files.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Character:
    """A roleplay character definition."""

    name: str
    persona: str  # Who the character is
    background: str  # Character's backstory
    greeting: str  # First message when starting a conversation
    traits: list[str] = field(default_factory=list)
    speech_style: str = "default"
    temperature: float = 0.8

    def build_system_prompt(self) -> str:
        """Build the full system prompt from character attributes."""
        traits_str = ", ".join(self.traits) if self.traits else "adaptable"

        return f"""You are roleplaying as {self.name}. Stay in character at all times.

## Character
{self.persona}

## Background
{self.background}

## Personality Traits
{traits_str}

## Speech Style
{self.speech_style}

## Rules
- Always respond as {self.name}, never break character.
- Use actions wrapped in *asterisks* to describe physical actions and emotions.
- React naturally to the user's messages based on your personality.
- Keep responses concise but immersive (2-4 paragraphs max).
- Never mention that you are an AI, language model, or assistant.
- If the user says something out of character, gently redirect in-character."""


# === Load Characters ===

def load_characters() -> dict[str, Character]:
    """Load characters from characters.json file."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    json_path = base_dir / "characters.json"
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                key: Character(**char_data)
                for key, char_data in data.items()
            }
    except FileNotFoundError:
        print(f"Warning: {json_path} not found.")
        return {}
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {}


CHARACTERS: dict[str, Character] = load_characters()


def get_character(name: str) -> Character | None:
    """Get a character by name (case-insensitive)."""
    return CHARACTERS.get(name.lower())


def list_characters() -> list[str]:
    """List all available character names."""
    return list(CHARACTERS.keys())
