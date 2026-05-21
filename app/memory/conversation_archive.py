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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get embedding vectors for a list of texts in a single batch request.

    Uses the configured embed model.
    """
    if not texts:
        return []
    resp = httpx.post(
        f"{config.ollama_base_url}/api/embed",
        json={"model": config.ollama_embed_model, "input": texts},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...vector1...], [...vector2...]]}
    return data["embeddings"]


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


def search_archive(
    client: QdrantClient,
    query: str,
    character_name: str,
    thread_id: str = "",
    top_k: int = 3,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    Search conversation archive for exchanges relevant to the query.

    Args:
        client: Qdrant client
        query: The user's current message to find relevant past context
        character_name: Filter to same character's conversations
        thread_id: Optional thread filter to keep conversations isolated
        top_k: Maximum number of results
        score_threshold: Minimum similarity score (0-1, cosine)

    Returns:
        List of dicts with 'user_message', 'ai_response', 'score'
    """
    try:
        _ensure_collection(client)
    except Exception as e:
        logger.warning("Failed to ensure Qdrant collection: %s", e)
        return []

    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        logger.warning("Failed to embed query: %s", e)
        return []

    filters = [
        FieldCondition(
            key="character_name",
            match=MatchValue(value=character_name.lower()),
        )
    ]
    if thread_id:
        filters.append(
            FieldCondition(
                key="thread_id",
                match=MatchValue(value=thread_id),
            )
        )

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=Filter(must=filters),
            limit=top_k,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.warning("Failed to search archive: %s", e)
        return []

    archive_results = []
    for point in results.points:
        payload = point.payload or {}
        if "user_message" in payload and "ai_response" in payload:
            archive_results.append(
                {
                    "user_message": payload["user_message"],
                    "ai_response": payload["ai_response"],
                    "score": point.score,
                }
            )
            continue

        for ex in payload.get("exchanges", []):
            if "user_message" not in ex or "ai_response" not in ex:
                continue
            archive_results.append(
                {
                    "user_message": ex["user_message"],
                    "ai_response": ex["ai_response"],
                    "score": point.score,
                }
            )

    return archive_results


def index_batch(
    client: QdrantClient,
    messages: list,
    character_name: str,
    thread_id: str = "",
):
    """
    Index a batch of conversation messages as a single Qdrant point.

    Extracts all user+AI exchange pairs from the message list,
    combines them into one text, embeds once, and stores as one vector.
    """
    try:
        _ensure_collection(client)
    except Exception as e:
        logger.warning("Failed to ensure Qdrant collection: %s", e)
        return

    # Extract exchanges
    exchanges = []
    i = 0
    while i < len(messages) - 1:
        if messages[i].type == "human" and messages[i + 1].type == "ai":
            exchanges.append(
                (messages[i].content, messages[i + 1].content)
            )
            i += 2
        else:
            i += 1

    if not exchanges:
        return

    # Combine all exchanges into one text
    combined_parts = []
    for user_msg, ai_resp in exchanges:
        combined_parts.append(f"User: {user_msg}\n{character_name}: {ai_resp}")
    combined_text = "\n---\n".join(combined_parts)

    point_id = _make_point_id(combined_text, character_name, thread_id)

    try:
        embedding = get_embedding(combined_text)
    except Exception as e:
        logger.warning("Failed to embed batch: %s", e)
        return

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "exchanges": [
                {"user_message": u, "ai_response": a} for u, a in exchanges
            ],
            "character_name": character_name.lower(),
            "thread_id": thread_id,
            "combined_text": combined_text,
            "indexed_at": datetime.now().isoformat(),
        },
    )

    try:
        client.upsert(collection_name=COLLECTION_NAME, points=[point])
    except Exception as e:
        logger.warning("Failed to upsert batch: %s", e)
        return

    logger.info("Indexed batch of %d exchanges into Qdrant", len(exchanges))
