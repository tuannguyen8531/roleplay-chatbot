"""Pydantic schemas for the API."""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    mongo: str
    qdrant: str
    provider: str


class CharacterSummary(BaseModel):
    slug: str
    name: str
    background: str


class CharacterDetail(CharacterSummary):
    persona: str
    greeting: str
    traits: list[str]
    speech_style: str
    temperature: float


class CreateThreadRequest(BaseModel):
    character_slug: str = Field(..., min_length=1)


class ThreadResponse(BaseModel):
    thread_id: str
    character_slug: str
    character_name: str
    greeting: str | None = None
    provider: str | None = None
    model: str | None = None
    turn_count: int = 0
    message_count: int = 0
    conversation_summary: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    character: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    thread_id: str | None = None


class SendMessageResponse(BaseModel):
    thread_id: str
    character_slug: str
    character_name: str
    response: str
    turn_count: int


class DeleteThreadResponse(BaseModel):
    thread_id: str
    deleted: bool
    mongo_records_deleted: int
    qdrant_deleted: bool
