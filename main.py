"""
AI Roleplay Chatbot — Terminal REPL

A local AI-powered roleplay chatbot using Ollama + LangGraph.
Run: uv run python main.py
"""

import atexit
import logging
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from app.config import config
from app.graph.builder import build_graph
from app.graph.nodes.diary import diary_node, should_run_diary
from app.models.character import Character, get_character, list_characters

# ── ANSI Colors ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"


class Spinner:
    """Animated spinner for terminal — shows while AI is thinking."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _animate(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r{DIM}{frame} {self.message}...{RESET}", end="", flush=True)
            i += 1
            self._stop_event.wait(0.1)
        # Clear the spinner line
        print(f"\r{' ' * (len(self.message) + 10)}\r", end="", flush=True)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def print_banner():
    """Print the startup banner."""
    print(f"""
{MAGENTA}{BOLD}╔══════════════════════════════════════════════════════╗
║           🎭  AI Roleplay Chatbot  🎭                ║
║       Ollama + LangGraph · Phase 2 (MongoDB)         ║
╚══════════════════════════════════════════════════════╝{RESET}
{DIM}Model: {config.ollama_model} · Temp: {config.ollama_temperature} · Ctx: {config.ollama_num_ctx}{RESET}
""")


def select_character() -> Character:
    """Interactive character selection."""
    characters = list_characters()

    print(f"{YELLOW}{BOLD}Available Characters:{RESET}")
    for i, name in enumerate(characters, 1):
        char = get_character(name)
        if char:
            print(f"  {CYAN}{i}.{RESET} {BOLD}{char.name}{RESET} — {DIM}{char.persona[:60]}...{RESET}")

    print()
    while True:
        choice = input(f"{YELLOW}Select character (1-{len(characters)}) or name: {RESET}").strip()

        # Try as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(characters):
                char = get_character(characters[idx])
                if char:
                    return char
        except ValueError:
            pass

        # Try as name
        char = get_character(choice)
        if char:
            return char

        print(f"{RED}Unknown character. Try again.{RESET}")


def print_character_greeting(character: Character):
    """Display the character's greeting message."""
    print(f"\n{DIM}{'─' * 56}{RESET}")
    print(f"{GREEN}{BOLD}{character.name}:{RESET} {character.greeting}")
    print(f"{DIM}{'─' * 56}{RESET}")
    print(f"{DIM}Commands: 'quit' to exit · 'switch' to change character · 'clear' to start new thread{RESET}\n")


def delete_thread(character: Character, mongo_client: MongoClient, thread_id: str, qdrant_client=None):
    """Delete all checkpoints and vector embeddings associated with a thread_id."""
    db = mongo_client[config.mongodb_db_name]
    
    # 1. Delete from MongoDB
    checkpoint_del = db["checkpoints"].delete_many({"thread_id": thread_id})
    writes_del = db["checkpoint_writes"].delete_many({"thread_id": thread_id})
    blobs_del = db["checkpoint_blobs"].delete_many({"thread_id": thread_id})
    total_mongo = checkpoint_del.deleted_count + writes_del.deleted_count + blobs_del.deleted_count

    # 2. Delete from Qdrant
    qdrant_ok = False
    if qdrant_client is not None:
        try:
            from qdrant_client.http import models as qdrant_models
            from app.memory.conversation_archive import COLLECTION_NAME
            
            # Check if collection exists
            collections = [c.name for c in qdrant_client.get_collections().collections]
            if COLLECTION_NAME in collections:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="thread_id",
                                    match=qdrant_models.MatchValue(value=thread_id),
                                )
                            ]
                        )
                    )
                )
                qdrant_ok = True
        except Exception as e:
            print(f"{YELLOW}⚠ Warning: Failed to delete vectors from Qdrant: {e}{RESET}")

    print(f"\n{GREEN}✓ Deleted session {BOLD}{thread_id}{RESET} ({total_mongo} MongoDB records cleared" + 
          (f", Qdrant archive cleared" if qdrant_ok else "") + ").")


def select_thread(character: Character, mongo_client: MongoClient, qdrant_client=None) -> str | None:
    """
    Let user choose to resume an existing thread, start a new one, or delete one.

    Returns thread_id string, or None for a new thread.
    """
    db = mongo_client[config.mongodb_db_name]
    checkpoints = db["checkpoints"]
    prefix = character.name.lower()

    while True:
        # Find distinct thread_ids that match this character
        pipeline = [
            {"$match": {"thread_id": {"$regex": f"^{prefix}-"}}},
            {"$sort": {"checkpoint_id": -1}},
            {"$group": {
                "_id": "$thread_id",
                "last_update": {"$first": "$checkpoint_id"},
                "step": {"$first": "$metadata.step"},
            }},
            {"$sort": {"last_update": -1}},
            {"$limit": 10},
        ]
        threads = list(checkpoints.aggregate(pipeline))

        if not threads:
            return None  # No existing threads, start new

        print(f"\n{YELLOW}{BOLD}Existing conversations:{RESET}")
        print(f"  {CYAN}0.{RESET} {BOLD}New conversation{RESET}")
        for i, t in enumerate(threads, 1):
            steps = t.get("step", "?")
            print(f"  {CYAN}{i}.{RESET} {t['_id']} {DIM}({steps} steps){RESET}")

        print()
        choice = input(
            f"{YELLOW}Select (0 for new, 1-{len(threads)} to resume, d<number> to delete, e.g. d1): {RESET}"
        ).strip()

        if not choice:
            continue

        # Check if delete command (e.g. d1 or d 1 or del1)
        is_delete = False
        idx_str = choice
        if choice.lower().startswith("d"):
            is_delete = True
            if choice.lower().startswith("del"):
                idx_str = choice[3:].strip()
            else:
                idx_str = choice[1:].strip()
        
        try:
            idx = int(idx_str)
            if is_delete:
                if 1 <= idx <= len(threads):
                    target_thread = threads[idx - 1]["_id"]
                    delete_thread(character, mongo_client, target_thread, qdrant_client)
                    # Loop will re-query and re-display updated list
                    continue
                else:
                    print(f"{RED}Invalid thread number to delete.{RESET}")
            else:
                if idx == 0:
                    return None
                if 1 <= idx <= len(threads):
                    return threads[idx - 1]["_id"]
        except ValueError:
            pass

        print(f"{RED}Invalid input. Use 0, a number, or d<number> to delete.{RESET}")


def run_chat(
    character: Character,
    checkpointer: MongoDBSaver,
    thread_id: str,
    bg_executor: ThreadPoolExecutor | None = None,
    qdrant_client=None,
):
    """
    Main chat loop for a single character session.

    Uses LangGraph with MongoDB checkpointer for persistent conversations.
    """
    # Build the graph with checkpointer
    graph = build_graph(checkpointer=checkpointer)

    graph_config = {"configurable": {"thread_id": thread_id}}

    # Check if resuming — load existing state
    existing_state = graph.get_state(graph_config)
    is_resume = existing_state.values.get("messages", []) != []

    if is_resume:
        messages = existing_state.values["messages"]
        print(f"\n{DIM}{'─' * 56}{RESET}")
        print(f"{GREEN}↻ Resuming thread: {thread_id} ({len(messages)} messages){RESET}")
        # Show last few messages for context
        recent = messages[-4:]  # last 2 exchanges
        for msg in recent:
            role = msg.type
            if role == "human":
                print(f"  {CYAN}You:{RESET} {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")
            else:
                print(f"  {GREEN}{character.name}:{RESET} {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")
        print(f"{DIM}{'─' * 56}{RESET}\n")
    else:
        print_character_greeting(character)

    print(f"{DIM}Thread: {thread_id}{RESET}\n")

    while True:
        # Get user input
        try:
            user_input = input(f"{CYAN}{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye!{RESET}")
            return "quit"

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "quit":
            print(f"\n{DIM}Farewell, traveler.{RESET}")
            return "quit"
        elif user_input.lower() == "switch":
            return "switch"
        elif user_input.lower() == "clear":
            # Start a new thread (old one remains in MongoDB)
            thread_id = f"{character.name.lower()}-{uuid.uuid4().hex[:8]}"
            graph_config = {"configurable": {"thread_id": thread_id}}
            print(f"{DIM}New conversation started.{RESET}")
            print(f"{DIM}Thread: {thread_id}{RESET}\n")
            print_character_greeting(character)
            continue

        # Invoke the graph with spinner
        try:
            with Spinner(f"{character.name} is thinking"):
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "character_name": character.name,
                        "character_prompt": character.build_system_prompt(),
                        "thread_id": thread_id,
                    },
                    config=graph_config,
                )

            # Extract AI response from result
            ai_message = result["messages"][-1]
            print(f"{GREEN}{BOLD}{character.name}:{RESET} {ai_message.content}")

            # Background tasks after response is shown
            if bg_executor is not None:
                # Trigger diary extraction if due
                if should_run_diary(result) == "diary":
                    state_snapshot = {
                        "messages": list(result.get("messages", [])),
                        "character_name": result.get("character_name", ""),
                        "long_term_facts": list(result.get("long_term_facts", [])),
                        "turn_count": result.get("turn_count", 0),
                    }
                    bg_executor.submit(_run_diary_safe, state_snapshot)

        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            print(f"{DIM}Make sure Ollama is running: ollama serve{RESET}")

        print()


def _run_diary_safe(state_snapshot: dict):
    """Safely run diary extraction in a background thread."""
    try:
        diary_node(state_snapshot)
    except Exception:
        logging.getLogger("diary").exception("Background diary extraction failed")


def main():
    """Main entry point."""
    print_banner()

    # Check if Gemini is configured
    if config.llm_provider.lower() == "gemini":
        print(f"{DIM}Using Gemini provider with model {config.gemini_model}...{RESET}")
        api_key = config.gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"{RED}✗ Gemini API key is missing. Set GEMINI_API_KEY in .env or GOOGLE_API_KEY in environment.{RESET}")
            sys.exit(1)
        print(f"{GREEN}✓ Gemini configuration checked (API key present).{RESET}")

    # Check if Ollama is reachable
    print(f"{DIM}Connecting to Ollama at {config.ollama_base_url}...{RESET}")
    ollama_ok = False
    try:
        resp = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            print(f"{GREEN}✓ Ollama connected. Available models: {', '.join(models)}{RESET}")
            ollama_ok = True
        else:
            print(f"{YELLOW}⚠ Ollama connected but no models found. Run: ollama pull {config.ollama_model}{RESET}")
            if config.llm_provider.lower() == "ollama":
                sys.exit(1)

        if ollama_ok and config.llm_provider.lower() == "ollama":
            if not any(config.ollama_model in m for m in models):
                print(f"{YELLOW}⚠ Model '{config.ollama_model}' not found. Run: ollama pull {config.ollama_model}{RESET}")
                sys.exit(1)

        # Also check embedding model
        if ollama_ok:
            if not any(config.ollama_embed_model in m for m in models):
                print(f"{YELLOW}⚠ Embedding model '{config.ollama_embed_model}' not found. Run: ollama pull {config.ollama_embed_model}{RESET}")
                print(f"  {DIM}Conversation archive will be disabled until the model is available.{RESET}")
    except (httpx.ConnectError, httpx.TimeoutException, Exception) as e:
        if config.llm_provider.lower() == "ollama":
            print(f"{RED}✗ Cannot connect to Ollama ({e}). Make sure it's running:{RESET}")
            print(f"  {DIM}ollama serve{RESET}")
            sys.exit(1)
        else:
            print(f"{YELLOW}⚠ Ollama not reachable ({e}). Local conversation archive and embedding features disabled.{RESET}")

    # Connect to MongoDB
    print(f"{DIM}Connecting to MongoDB...{RESET}")
    try:
        mongo_client = MongoClient(config.mongodb_uri)
        # Verify connection is alive
        mongo_client.admin.command("ping")
        checkpointer = MongoDBSaver(mongo_client, db_name=config.mongodb_db_name)

        # Share mongo_client with graph nodes that need it
        from app.graph.nodes.entry import set_mongo_client as entry_set_mongo
        from app.graph.nodes.diary import set_mongo_client as diary_set_mongo
        entry_set_mongo(mongo_client)
        diary_set_mongo(mongo_client)

        print(f"{GREEN}✓ MongoDB connected. DB: {config.mongodb_db_name}{RESET}")
    except Exception as e:
        print(f"{RED}✗ Cannot connect to MongoDB: {e}{RESET}")
        print(f"  {DIM}Run: docker compose up -d{RESET}")
        sys.exit(1)

    # Connect to Qdrant
    print(f"{DIM}Connecting to Qdrant at {config.qdrant_url}...{RESET}")
    qdrant_client = None
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=config.qdrant_url, timeout=5)
        # Verify connection
        qdrant_client.get_collections()

        # Share qdrant_client with entry node
        from app.graph.nodes.entry import set_qdrant_client
        set_qdrant_client(qdrant_client)

        print(f"{GREEN}✓ Qdrant connected.{RESET}")
    except Exception as e:
        print(f"{YELLOW}⚠ Qdrant not available ({e}). Conversation archive disabled.{RESET}")
        print(f"  {DIM}Run: docker compose up -d{RESET}")

    print()

    # Thread pool for background tasks (diary + archive indexing)
    bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bg")

    # Share bg_executor with entry node
    from app.graph.nodes.entry import set_bg_executor
    set_bg_executor(bg_executor)

    def _shutdown_bg():
        print(f"\n{DIM}Waiting for background tasks...{RESET}")
        bg_executor.shutdown(wait=True)

    atexit.register(_shutdown_bg)

    # Main loop: select character → chat → repeat
    while True:
        character = select_character()
        print(f"\n{GREEN}✓ Playing as {BOLD}{character.name}{RESET}")

        # Let user resume existing thread or start new
        existing_thread = select_thread(character, mongo_client, qdrant_client=qdrant_client)
        thread_id = existing_thread or f"{character.name.lower()}-{uuid.uuid4().hex[:8]}"

        result = run_chat(character, checkpointer, thread_id, bg_executor, qdrant_client)

        if result == "quit":
            break
        # result == "switch" → loop back to character selection

    print(f"\n{MAGENTA}{BOLD}Thanks for playing! 🎭{RESET}\n")


if __name__ == "__main__":
    main()
