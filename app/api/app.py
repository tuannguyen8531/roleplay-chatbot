"""FastAPI entrypoint for the roleplay chatbot."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query, Request
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from app.config import config
from app.messages import message_content_to_text
from app.models.character import Character, get_character, list_characters
from app.runtime import (
    connect_mongo,
    connect_qdrant,
    create_thread_document,
    delete_thread,
    get_graph_state,
    get_thread_character,
    list_thread_documents,
    message_to_dict,
    public_config,
    run_chat_turn,
    setup_background_executor,
    shutdown_background_executor,
    touch_thread_document,
)

from app.api.schemas import (
    CharacterDetail,
    CharacterSummary,
    ChatRequest,
    ChatMessage,
    CreateThreadRequest,
    DeleteThreadResponse,
    HealthResponse,
    SendMessageRequest,
    SendMessageResponse,
    ThreadResponse,
)


@dataclass
class ApiRuntime:
    mongo_client: MongoClient
    checkpointer: MongoDBSaver
    bg_executor: ThreadPoolExecutor
    qdrant_client: object | None = None


def _preview(text: str, limit: int = 96) -> str:
    value = text.replace("\n", " ")
    return f"{value[: limit - 3]}..." if len(value) > limit else value


def _runtime(request: Request) -> ApiRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="API runtime is not ready")
    return runtime


def _character_or_404(slug: str) -> Character:
    character = get_character(slug)
    if character is None:
        raise HTTPException(status_code=404, detail=f"Character not found: {slug}")
    return character


def _thread_response_from_doc(doc: dict, greeting: str | None = None) -> ThreadResponse:
    return ThreadResponse(
        thread_id=doc["thread_id"],
        character_slug=doc.get("character_slug", ""),
        character_name=doc.get("character_name", ""),
        greeting=greeting,
        provider=doc.get("provider"),
        model=doc.get("model"),
        turn_count=doc.get("turn_count", 0),
        message_count=doc.get("message_count", 0),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


def _thread_response_from_state(thread_id: str, runtime: ApiRuntime) -> ThreadResponse:
    state = get_graph_state(thread_id, runtime.checkpointer)
    if not state.values:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    character_name = state.values.get("character_name", "")
    character = get_character(character_name) if character_name else None
    messages = state.values.get("messages", [])
    return ThreadResponse(
        thread_id=thread_id,
        character_slug=character.slug if character else "",
        character_name=character.name if character else character_name,
        provider=config.llm_provider,
        model=config.active_model,
        turn_count=state.values.get("turn_count", 0),
        message_count=len(messages),
        conversation_summary=state.values.get("conversation_summary", ""),
    )


def _assert_thread_belongs_to_character(
    *,
    runtime: ApiRuntime,
    thread_id: str,
    character: Character,
) -> None:
    """Ensure an existing thread belongs to the requested character."""
    doc = runtime.mongo_client[config.mongodb_db_name]["threads"].find_one(
        {"thread_id": thread_id}
    )
    if doc is not None:
        if doc.get("character_slug") != character.slug:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Thread {thread_id} belongs to character "
                    f"{doc.get('character_slug')}, not {character.slug}"
                ),
            )
        return

    thread_character = get_thread_character(thread_id, runtime.checkpointer)
    if thread_character is None:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")
    if thread_character.slug != character.slug:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Thread {thread_id} belongs to character "
                f"{thread_character.slug}, not {character.slug}"
            ),
        )


def _run_message_for_character(
    *,
    runtime: ApiRuntime,
    character: Character,
    thread_id: str,
    message: str,
) -> SendMessageResponse:
    """Run one message and serialize the API response."""
    result = run_chat_turn(
        character=character,
        thread_id=thread_id,
        message=message,
        checkpointer=runtime.checkpointer,
        bg_executor=runtime.bg_executor,
    )
    touch_thread_document(
        runtime.mongo_client,
        character,
        thread_id,
        result=cast(dict[str, Any], result),
    )
    ai_message = result["messages"][-1]
    return SendMessageResponse(
        thread_id=thread_id,
        character_slug=character.slug,
        character_name=character.name,
        response=message_content_to_text(ai_message.content),
        turn_count=result.get("turn_count", 0),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_client, checkpointer = connect_mongo()
    qdrant_client = None
    try:
        qdrant_client = connect_qdrant()
    except Exception:
        qdrant_client = None

    bg_executor = setup_background_executor(register_atexit=False)
    app.state.runtime = ApiRuntime(
        mongo_client=mongo_client,
        checkpointer=checkpointer,
        bg_executor=bg_executor,
        qdrant_client=qdrant_client,
    )
    try:
        yield
    finally:
        shutdown_background_executor(bg_executor)
        mongo_client.close()


app = FastAPI(
    title="Roleplay Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    runtime = _runtime(request)
    return HealthResponse(
        status="ok",
        mongo="ok",
        qdrant="ok" if runtime.qdrant_client is not None else "disabled",
        provider=config.llm_provider,
    )


@app.get("/config")
def get_config():
    return public_config()


@app.post("/chat", response_model=SendMessageResponse)
def chat(payload: ChatRequest, request: Request):
    runtime = _runtime(request)
    character = _character_or_404(payload.character)

    if payload.thread_id:
        thread_id = payload.thread_id
        _assert_thread_belongs_to_character(
            runtime=runtime,
            thread_id=thread_id,
            character=character,
        )
    else:
        doc = create_thread_document(runtime.mongo_client, character)
        thread_id = doc["thread_id"]

    return _run_message_for_character(
        runtime=runtime,
        character=character,
        thread_id=thread_id,
        message=payload.message,
    )


@app.get("/characters", response_model=list[CharacterSummary])
def get_characters():
    characters = []
    for slug in list_characters():
        character = get_character(slug)
        if character is None:
            continue
        characters.append(
            CharacterSummary(
                slug=slug,
                name=character.name,
                background=_preview(character.background),
            )
        )
    return characters


@app.get("/characters/{slug}", response_model=CharacterDetail)
def get_character_detail(slug: str):
    character = _character_or_404(slug)
    return CharacterDetail(
        slug=character.slug,
        name=character.name,
        background=character.background,
        persona=character.persona,
        greeting=character.greeting,
        traits=character.traits,
        speech_style=character.speech_style,
        temperature=character.temperature,
    )


@app.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(payload: CreateThreadRequest, request: Request):
    runtime = _runtime(request)
    character = _character_or_404(payload.character_slug)
    doc = create_thread_document(runtime.mongo_client, character)
    return _thread_response_from_doc(doc, greeting=character.greeting)


@app.get("/threads", response_model=list[ThreadResponse])
def get_threads(
    request: Request,
    character_slug: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
):
    runtime = _runtime(request)
    character = _character_or_404(character_slug) if character_slug else None
    docs = list_thread_documents(runtime.mongo_client, character=character, limit=limit)
    return [_thread_response_from_doc(doc) for doc in docs]


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(thread_id: str, request: Request):
    runtime = _runtime(request)
    doc = runtime.mongo_client[config.mongodb_db_name]["threads"].find_one(
        {"thread_id": thread_id}
    )
    if doc:
        response = _thread_response_from_doc(doc)
        try:
            state_response = _thread_response_from_state(thread_id, runtime)
            response.turn_count = state_response.turn_count
            response.message_count = state_response.message_count
            response.conversation_summary = state_response.conversation_summary
        except HTTPException:
            pass
        return response
    return _thread_response_from_state(thread_id, runtime)


@app.get("/threads/{thread_id}/messages", response_model=list[ChatMessage])
def get_thread_messages(thread_id: str, request: Request):
    runtime = _runtime(request)
    state = get_graph_state(thread_id, runtime.checkpointer)
    if not state.values:
        doc = runtime.mongo_client[config.mongodb_db_name]["threads"].find_one(
            {"thread_id": thread_id}
        )
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")
        return []
    return [ChatMessage(**message_to_dict(message)) for message in state.values.get("messages", [])]


@app.post("/threads/{thread_id}/messages", response_model=SendMessageResponse)
def send_message(thread_id: str, payload: SendMessageRequest, request: Request):
    runtime = _runtime(request)
    character = get_thread_character(thread_id, runtime.checkpointer)
    if character is None:
        doc = runtime.mongo_client[config.mongodb_db_name]["threads"].find_one(
            {"thread_id": thread_id}
        )
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")
        character = _character_or_404(doc["character_slug"])

    return _run_message_for_character(
        runtime=runtime,
        character=character,
        thread_id=thread_id,
        message=payload.message,
    )


@app.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
def remove_thread(thread_id: str, request: Request):
    runtime = _runtime(request)
    result = delete_thread(thread_id, runtime.mongo_client, runtime.qdrant_client)
    return DeleteThreadResponse(
        thread_id=thread_id,
        deleted=result["mongo_records_deleted"] > 0 or result["qdrant_deleted"],
        mongo_records_deleted=result["mongo_records_deleted"],
        qdrant_deleted=result["qdrant_deleted"],
    )
