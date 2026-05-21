"""
Facts Store — MongoDB-backed long-term memory for cross-session facts.

Stores extracted facts (user preferences, plot points, character knowledge)
in a dedicated MongoDB collection, separate from checkpoints.
"""

from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import config


def _get_collection(client: MongoClient) -> Collection:
    """Get the facts collection."""
    db = client[config.mongodb_db_name]
    return db["user_facts"]


def load_facts(client: MongoClient, character_name: str, user_id: str = "default") -> list[str]:
    """
    Load long-term facts from MongoDB for a specific user + character pair.

    Args:
        client: MongoDB client
        character_name: Name of the character (facts are per-character)
        user_id: User identifier (for multi-user support later)

    Returns:
        List of fact strings
    """
    collection = _get_collection(client)
    doc = collection.find_one({
        "user_id": user_id,
        "character_name": character_name.lower(),
    })
    if doc:
        return doc.get("facts", [])
    return []


def save_facts(client: MongoClient, facts: list[str], character_name: str, user_id: str = "default"):
    """
    Save/update long-term facts to MongoDB.

    Uses upsert — creates document if it doesn't exist, updates if it does.
    Uses atomic $addToSet to avoid race conditions between background threads.
    """
    collection = _get_collection(client)

    # Filter out empty strings
    stripped_facts = [f.strip() for f in facts if f.strip()]
    if not stripped_facts:
        return []

    collection.update_one(
        {"user_id": user_id, "character_name": character_name.lower()},
        {"$addToSet": {"facts": {"$each": stripped_facts}}},
        upsert=True,
    )

    # Return updated list
    return load_facts(client, character_name, user_id)
