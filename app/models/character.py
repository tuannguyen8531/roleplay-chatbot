"""
Character data model and default character definitions.

MongoDB is the primary character store at runtime. characters.json remains the
seed/default source for local development and first startup.
"""

import json
import re
from datetime import datetime, timezone
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
    slug: str = ""

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


# === Character Store ===


_CHARACTER_COLLECTION = "characters"
_CHARACTER_FIELDS = {
    "name",
    "persona",
    "background",
    "greeting",
    "traits",
    "speech_style",
    "temperature",
    "slug",
}


def _default_slug(name: str) -> str:
    """Build a stable lowercase slug from a name or JSON key."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or name.lower()


def _json_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "characters.json"


def _character_from_mapping(data: dict, slug: str = "") -> Character:
    """Create a Character while ignoring Mongo metadata fields."""
    char_data = {key: data[key] for key in _CHARACTER_FIELDS if key in data}
    char_data["slug"] = slug or char_data.get("slug") or _default_slug(char_data["name"])
    return Character(**char_data)


def _character_to_document(slug: str, character: Character) -> dict:
    return {
        "slug": slug,
        "name": character.name,
        "persona": character.persona,
        "background": character.background,
        "greeting": character.greeting,
        "traits": character.traits,
        "speech_style": character.speech_style,
        "temperature": character.temperature,
        "is_active": True,
        "version": 1,
    }


def load_json_characters() -> dict[str, Character]:
    """Load default characters from characters.json file."""
    json_path = _json_path()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                key.lower(): _character_from_mapping(char_data, slug=key.lower())
                for key, char_data in data.items()
            }
    except FileNotFoundError:
        print(f"Warning: {json_path} not found.")
        return {}
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {}


def load_characters() -> dict[str, Character]:
    """Backward-compatible loader for default JSON characters."""
    return load_json_characters()


def _load_mongo_characters(collection) -> dict[str, Character]:
    docs = collection.find({"is_active": {"$ne": False}}).sort("slug", 1)
    characters: dict[str, Character] = {}
    for doc in docs:
        slug = doc.get("slug") or _default_slug(doc["name"])
        characters[slug] = _character_from_mapping(doc, slug=slug)
    return characters


def initialize_character_store(mongo_client, db_name: str, seed_defaults: bool = True) -> int:
    """
    Initialize MongoDB character storage and refresh the in-memory cache.

    Existing Mongo characters are preserved. JSON defaults are inserted only
    when their slug is missing, so local edits in Mongo are not overwritten.
    """
    db = mongo_client[db_name]
    collection = db[_CHARACTER_COLLECTION]
    collection.create_index("slug", unique=True)
    collection.create_index([("is_active", 1), ("slug", 1)])

    if seed_defaults:
        now = datetime.now(timezone.utc)
        for slug, character in load_json_characters().items():
            document = _character_to_document(slug, character)
            document["created_at"] = now
            document["updated_at"] = now
            collection.update_one(
                {"slug": slug},
                {"$setOnInsert": document},
                upsert=True,
            )

    characters = _load_mongo_characters(collection)
    if not characters:
        characters = load_json_characters()

    global CHARACTERS
    CHARACTERS = characters
    return len(CHARACTERS)


CHARACTERS: dict[str, Character] = load_json_characters()


def get_character(name: str) -> Character | None:
    """Get a character by slug or display name (case-insensitive)."""
    normalized = name.strip().lower()
    character = CHARACTERS.get(normalized) or CHARACTERS.get(_default_slug(normalized))
    if character:
        return character

    for character in CHARACTERS.values():
        if character.name.lower() == normalized:
            return character
    return None


def list_characters() -> list[str]:
    """List all available character names."""
    return sorted(CHARACTERS.keys())
