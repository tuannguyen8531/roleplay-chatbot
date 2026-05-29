# Roleplay Chatbot

Local AI roleplay chatbot built with LangGraph, MongoDB, Qdrant, and either Ollama or Gemini.

The app is currently a terminal CLI. It persists conversation state in MongoDB, keeps long-term facts per character, and uses Qdrant plus local embeddings for recall/archive search.

## Features

- Roleplay as configurable characters.
- Persistent conversations through LangGraph MongoDB checkpoints.
- Character definitions loaded from MongoDB at runtime.
- `characters.json` is used as seed/default data for first startup.
- Long-term user facts stored in MongoDB.
- Conversation archive and semantic recall through Qdrant.
- Supports Ollama and Gemini providers.
- Separate request/response logs with matching `call_id` values.

## Requirements

- Python 3.12+
- `uv`
- Docker and Docker Compose
- Ollama, if using the `ollama` provider
- Gemini API key, if using the `gemini` provider

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file:

```env
LLM_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_UTILITY_MODEL=
OLLAMA_TEMPERATURE=0.8
OLLAMA_NUM_CTX=8192
OLLAMA_EMBED_MODEL=nomic-embed-text

GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-flash
GEMINI_UTILITY_MODEL=gemini-1.5-flash

MONGODB_USER=chatbot
MONGODB_PASSWORD=chatbot
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB_NAME=chatbot

QDRANT_HOST=localhost
QDRANT_PORT=6333

MAX_HISTORY_MESSAGES=20
DIARY_INTERVAL=5
STREAMING=true
RAG_SCORE_THRESHOLD=0.55
RAG_MIN_MESSAGE_LENGTH=15
```

Start MongoDB and Qdrant:

```bash
docker compose up -d
```

If using Ollama, start Ollama and pull the required models:

```bash
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Run

Start the interactive CLI:

```bash
uv run python main.py
```

List characters:

```bash
uv run python main.py --list
```

Start a new thread with a character:

```bash
uv run python main.py --new aria
```

Resume the latest thread for a character:

```bash
uv run python main.py --resume aria
```

Resume a specific thread:

```bash
uv run python main.py --thread aria-1234abcd
```

Override the provider for one run:

```bash
uv run python main.py --provider gemini --new aria
```

Print the effective config:

```bash
uv run python main.py --config
```

## CLI Flags

```text
-p, --provider ollama|gemini   Override LLM_PROVIDER from environment
-c, --config                   Print effective config and exit
-l, --list                     List available characters and exit
-n, --new <slug>               Start a new thread for a character
-r, --resume <slug>            Resume latest character thread, or create one
-t, --thread <id>              Resume a specific thread id
```

`--new`, `--resume`, and `--thread` are mutually exclusive.

## Runtime Commands

Inside the chat CLI:

```text
quit    Exit the app
switch  Return to character selection
clear   Start a new thread for the current character
```

## Character Storage

Characters are stored in MongoDB collection `characters`.

On startup, the app:

1. Connects to MongoDB.
2. Creates indexes for the `characters` collection.
3. Seeds missing default characters from `characters.json`.
4. Loads active characters from MongoDB into runtime cache.

Existing MongoDB character documents are not overwritten by `characters.json`.

Typical character document:

```json
{
  "slug": "aria",
  "name": "Aria",
  "persona": "...",
  "background": "...",
  "greeting": "...",
  "traits": ["brave", "witty"],
  "speech_style": "...",
  "temperature": 0.85,
  "is_active": true,
  "version": 1
}
```
```

## Project Structure

```text
main.py                         CLI entrypoint
characters.json                 Default character seed data
docker-compose.yml              MongoDB and Qdrant services
app/config.py                   Environment-based configuration
app/llm.py                      LLM provider factory
app/logger.py                   Request/response logging
app/models/character.py         Character model and MongoDB store init
app/models/state.py             LangGraph state schema
app/graph/builder.py            LangGraph assembly
app/graph/nodes/entry.py        Facts, RAG, summarization, trimming
app/graph/nodes/generate.py     Main response generation
app/graph/nodes/diary.py        Long-term fact extraction
app/memory/facts_store.py       MongoDB fact storage
app/memory/conversation_archive.py  Qdrant archive and retrieval
```

## Notes

- MongoDB is required for normal CLI startup.
- Qdrant is optional at runtime; if unavailable, conversation archive recall is disabled.
- For Gemini provider, local Ollama embedding/archive features may be unavailable unless Ollama is also running.
- This project is currently prepared for an API direction, but no FastAPI app has been added yet.
