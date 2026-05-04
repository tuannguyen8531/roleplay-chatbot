"""
Conversation Archive — Vector-based memory for past conversation retrieval.

Uses Qdrant to store and retrieve past conversation exchanges.
Each exchange (user message + AI response) is embedded and indexed,
allowing the chatbot to recall specific details from old conversations
that may have been lost during summarization.

Embedding: Ollama's nomic-embed-text (runs locally, no API key needed)
"""

import hashlib
import logging
from datetime import datetime

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import config

logger = logging.getLogger("archive")

# Collection name in Qdrant
COLLECTION_NAME = "conversation_archive"

# Embedding dimension for nomic-embed-text (768d)
EMBED_DIM = 768


def get_embedding(text: str) -> list[float]:
    """
    Get embedding vector from Ollama's embedding API.

    Uses the configured embed model (default: nomic-embed-text).
    """
    resp = httpx.post(
        f"{config.ollama_base_url}/api/embed",
        json={"model": config.ollama_embed_model, "input": text},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...vector...]]}
    return data["embeddings"][0]


def _ensure_collection(client: QdrantClient):
    """Create the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", COLLECTION_NAME)


def _make_point_id(text: str, character: str, thread_id: str = "") -> str:
    """Generate a deterministic ID scoped to thread + character + content."""
    content = f"{thread_id}:{character}:{text}"
    return hashlib.md5(content.encode()).hexdigest()


def index_exchange(
    client: QdrantClient,
    user_message: str,
    ai_response: str,
    character_name: str,
    thread_id: str = "",
):
    """
    Index a single conversation exchange (user + AI) into Qdrant.

    The exchange is stored as a single point with the combined text
    as the content to embed, and metadata for filtering.
    """
    _ensure_collection(client)

    # Combine both sides of the exchange for richer semantic search
    combined_text = f"User: {user_message}\n{character_name}: {ai_response}"
    point_id = _make_point_id(combined_text, character_name, thread_id)

    try:
        embedding = get_embedding(combined_text)
    except Exception as e:
        logger.warning("Failed to embed exchange: %s", e)
        return

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "user_message": user_message,
            "ai_response": ai_response,
            "character_name": character_name.lower(),
            "thread_id": thread_id,
            "combined_text": combined_text,
            "indexed_at": datetime.now().isoformat(),
        },
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])


def search_archive(
    client: QdrantClient,
    query: str,
    character_name: str,
    top_k: int = 3,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    Search conversation archive for exchanges relevant to the query.

    Args:
        client: Qdrant client
        query: The user's current message to find relevant past context
        character_name: Filter to same character's conversations
        top_k: Maximum number of results
        score_threshold: Minimum similarity score (0-1, cosine)

    Returns:
        List of dicts with 'user_message', 'ai_response', 'score'
    """
    _ensure_collection(client)

    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        logger.warning("Failed to embed query: %s", e)
        return []

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="character_name",
                    match=MatchValue(value=character_name.lower()),
                )
            ]
        ),
        limit=top_k,
        score_threshold=score_threshold,
    )

    return [
        {
            "user_message": point.payload["user_message"],
            "ai_response": point.payload["ai_response"],
            "score": point.score,
        }
        for point in results.points
    ]


def index_messages(
    client: QdrantClient,
    messages: list,
    character_name: str,
    thread_id: str = "",
):
    """
    Index multiple messages as exchanges (pairs of user + AI).

    Iterates through messages and pairs each human message with
    the following AI response.
    """
    i = 0
    while i < len(messages) - 1:
        if messages[i].type == "human" and messages[i + 1].type == "ai":
            index_exchange(
                client,
                user_message=messages[i].content,
                ai_response=messages[i + 1].content,
                character_name=character_name,
                thread_id=thread_id,
            )
            i += 2
        else:
            i += 1
