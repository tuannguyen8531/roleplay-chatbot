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
    New facts are merged with existing ones (no duplicates).
    """
    collection = _get_collection(client)

    # Load existing facts to merge
    existing = load_facts(client, character_name, user_id)

    # Merge: add new facts that don't already exist
    merged = list(existing)
    for fact in facts:
        fact_stripped = fact.strip()
        if fact_stripped and fact_stripped not in merged:
            merged.append(fact_stripped)

    collection.update_one(
        {"user_id": user_id, "character_name": character_name.lower()},
        {"$set": {"facts": merged}},
        upsert=True,
    )

    return merged
