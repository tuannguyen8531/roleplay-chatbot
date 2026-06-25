"""Shared runtime helpers for CLI and API entrypoints."""

from __future__ import annotations

import atexit
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from app.config import config
from app.graph.builder import build_graph
from app.graph.nodes.diary import diary_node, should_run_diary
from app.messages import message_content_to_text
from app.models.character import Character, get_character, initialize_character_store
from app.models.state import RoleplayState


def make_thread_id(character: Character) -> str:
    """Create a new thread id for a character."""
    prefix = character.slug or character.name.lower()
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def public_config() -> dict:
    """Return config fields safe to expose over CLI/API."""
    data = {}
    for field in fields(config):
        name = field.name
        value = getattr(config, name)
        if "key" in name.lower() or "password" in name.lower():
            data[name] = bool(value)
        else:
            data[name] = value
    data["active_model"] = config.active_model
    data["utility_model"] = config.utility_model
    data["qdrant_url"] = config.qdrant_url
    return data


def connect_mongo() -> tuple[MongoClient, MongoDBSaver]:
    """Connect MongoDB, initialize shared node clients, and load characters."""
    mongo_client = MongoClient(config.mongodb_uri)
    mongo_client.admin.command("ping")
    checkpointer = MongoDBSaver(mongo_client, db_name=config.mongodb_db_name)

    from app.graph.nodes.entry import set_mongo_client as entry_set_mongo
    from app.graph.nodes.diary import set_mongo_client as diary_set_mongo

    entry_set_mongo(mongo_client)
    diary_set_mongo(mongo_client)
    initialize_character_store(mongo_client, db_name=config.mongodb_db_name)

    return mongo_client, checkpointer


def connect_qdrant():
    """Connect Qdrant if available and share the client with graph nodes."""
    from qdrant_client import QdrantClient

    qdrant_client = QdrantClient(url=config.qdrant_url, timeout=5)
    qdrant_client.get_collections()

    from app.graph.nodes.entry import set_qdrant_client

    set_qdrant_client(qdrant_client)
    return qdrant_client


def setup_background_executor(register_atexit: bool = True) -> ThreadPoolExecutor:
    """Create and register the shared background executor."""
    bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bg")

    from app.graph.nodes.entry import set_bg_executor

    set_bg_executor(bg_executor)

    if register_atexit:
        atexit.register(lambda: bg_executor.shutdown(wait=True))

    return bg_executor


def shutdown_background_executor(bg_executor: ThreadPoolExecutor | None):
    """Shutdown a background executor if present."""
    if bg_executor is not None:
        bg_executor.shutdown(wait=True)


def thread_prefixes(character: Character) -> list[str]:
    """Return known thread id prefixes for a character."""
    prefixes = []
    if character.slug:
        prefixes.append(character.slug.lower())
    prefixes.append(character.name.lower())
    return list(dict.fromkeys(prefixes))


def find_recent_threads(
    character: Character | None,
    mongo_client: MongoClient,
    limit: int = 20,
) -> list[dict]:
    """Find recent checkpoint thread ids, optionally filtered by character."""
    checkpoints = mongo_client[config.mongodb_db_name]["checkpoints"]
    match = {}
    if character is not None:
        prefixes = "|".join(re.escape(prefix) for prefix in thread_prefixes(character))
        match = {"thread_id": {"$regex": f"^({prefixes})-"}}

    pipeline = [
        {"$match": match},
        {"$sort": {"checkpoint_id": -1}},
        {"$group": {
            "_id": "$thread_id",
            "last_update": {"$first": "$checkpoint_id"},
            "step": {"$first": "$metadata.step"},
        }},
        {"$sort": {"last_update": -1}},
        {"$limit": limit},
    ]
    return list(checkpoints.aggregate(pipeline))


def get_latest_thread(character: Character, mongo_client: MongoClient) -> str | None:
    """Return the latest thread id for a character if one exists."""
    threads = find_recent_threads(character, mongo_client, limit=1)
    return threads[0]["_id"] if threads else None


def create_thread_document(
    mongo_client: MongoClient,
    character: Character,
    thread_id: str | None = None,
) -> dict:
    """Create or update API thread metadata."""
    now = datetime.now(timezone.utc)
    thread_id = thread_id or make_thread_id(character)
    document = {
        "thread_id": thread_id,
        "character_slug": character.slug,
        "character_name": character.name,
        "provider": config.llm_provider,
        "model": config.active_model,
        "created_at": now,
        "updated_at": now,
    }
    mongo_client[config.mongodb_db_name]["threads"].update_one(
        {"thread_id": thread_id},
        {"$setOnInsert": document},
        upsert=True,
    )
    return document


def touch_thread_document(
    mongo_client: MongoClient,
    character: Character,
    thread_id: str,
    result: RoleplayState | dict[str, Any] | None = None,
):
    """Refresh API thread metadata after a chat turn."""
    update = {
        "character_slug": character.slug,
        "character_name": character.name,
        "provider": config.llm_provider,
        "model": config.active_model,
        "updated_at": datetime.now(timezone.utc),
    }
    if result is not None:
        update["turn_count"] = result.get("turn_count", 0)
        update["message_count"] = len(result.get("messages", []))
    mongo_client[config.mongodb_db_name]["threads"].update_one(
        {"thread_id": thread_id},
        {
            "$set": update,
            "$setOnInsert": {
                "thread_id": thread_id,
                "created_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
    )


def list_thread_documents(
    mongo_client: MongoClient,
    character: Character | None = None,
    limit: int = 20,
) -> list[dict]:
    """List API-created thread metadata."""
    query = {}
    if character is not None:
        query["character_slug"] = character.slug
    cursor = (
        mongo_client[config.mongodb_db_name]["threads"]
        .find(query)
        .sort("updated_at", -1)
        .limit(limit)
    )
    return list(cursor)


def get_graph_state(thread_id: str, checkpointer: MongoDBSaver):
    """Load LangGraph state for a thread."""
    graph = build_graph(checkpointer=checkpointer)
    graph_config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    return graph.get_state(graph_config)


def get_thread_character(thread_id: str, checkpointer: MongoDBSaver) -> Character | None:
    """Resolve a thread's character from its checkpointed graph state."""
    state = get_graph_state(thread_id, checkpointer)
    character_name = state.values.get("character_name")
    if not character_name:
        return None
    return get_character(character_name)


def message_to_dict(message) -> dict:
    """Serialize a LangChain message for API responses."""
    role = "assistant" if message.type == "ai" else message.type
    return {
        "role": role,
        "content": message_content_to_text(message.content),
    }


def run_chat_turn(
    *,
    character: Character,
    thread_id: str,
    message: str,
    checkpointer: MongoDBSaver,
    bg_executor: ThreadPoolExecutor | None = None,
) -> RoleplayState:
    """Run one user message through the graph and return the resulting state."""
    graph = build_graph(checkpointer=checkpointer)
    graph_config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    graph_input = cast(
        RoleplayState,
        {
            "messages": [HumanMessage(content=message)],
            "character_name": character.name,
            "character_prompt": character.build_system_prompt(),
            "thread_id": thread_id,
        },
    )
    result = cast(
        RoleplayState,
        graph.invoke(
            graph_input,
            config=graph_config,
        ),
    )

    if bg_executor is not None and should_run_diary(result) == "diary":
        state_snapshot = cast(
            RoleplayState,
            {
                "messages": list(result.get("messages", [])),
                "character_name": result.get("character_name", ""),
                "long_term_facts": list(result.get("long_term_facts", [])),
                "turn_count": result.get("turn_count", 0),
            },
        )
        bg_executor.submit(_run_diary_safe, state_snapshot)

    return result


def _run_diary_safe(state_snapshot: RoleplayState):
    """Safely run diary extraction in a background thread."""
    try:
        diary_node(state_snapshot)
    except Exception:
        import logging

        logging.getLogger("diary").exception("Background diary extraction failed")


def delete_thread(thread_id: str, mongo_client: MongoClient, qdrant_client=None) -> dict:
    """Delete all checkpoints and vector embeddings associated with a thread_id."""
    db = mongo_client[config.mongodb_db_name]
    thread_doc_del = db["threads"].delete_many({"thread_id": thread_id})
    checkpoint_del = db["checkpoints"].delete_many({"thread_id": thread_id})
    writes_del = db["checkpoint_writes"].delete_many({"thread_id": thread_id})
    blobs_del = db["checkpoint_blobs"].delete_many({"thread_id": thread_id})
    total_mongo = (
        checkpoint_del.deleted_count
        + writes_del.deleted_count
        + blobs_del.deleted_count
    )

    qdrant_deleted = False
    if qdrant_client is not None:
        try:
            from qdrant_client.http import models as qdrant_models

            from app.memory.conversation_archive import COLLECTION_NAME

            collections = [c.name for c in qdrant_client.get_collections().collections]
            if COLLECTION_NAME in collections:
                filters: list[qdrant_models.Condition] = [
                    qdrant_models.FieldCondition(
                        key="thread_id",
                        match=qdrant_models.MatchValue(value=thread_id),
                    )
                ]
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(must=filters)
                    ),
                )
                qdrant_deleted = True
        except Exception:
            qdrant_deleted = False

    return {
        "thread_id": thread_id,
        "mongo_records_deleted": total_mongo + thread_doc_del.deleted_count,
        "qdrant_deleted": qdrant_deleted,
    }
