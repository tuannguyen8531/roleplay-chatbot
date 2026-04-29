"""
AI Roleplay Chatbot — Terminal REPL

A local AI-powered roleplay chatbot using Ollama + LangGraph.
Run: uv run python main.py
"""

import atexit
import logging
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


def select_thread(character: Character, mongo_client: MongoClient) -> str | None:
    """
    Let user choose to resume an existing thread or start a new one.

    Returns thread_id string, or None for a new thread.
    """
    # Query MongoDB for existing threads for this character
    db = mongo_client[config.mongodb_db_name]
    checkpoints = db["checkpoints"]

    # Find distinct thread_ids that match this character
    prefix = character.name.lower()
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
    while True:
        choice = input(f"{YELLOW}Select (0 for new, 1-{len(threads)} to resume): {RESET}").strip()
        try:
            idx = int(choice)
            if idx == 0:
                return None
            if 1 <= idx <= len(threads):
                return threads[idx - 1]["_id"]
        except ValueError:
            pass
        print(f"{RED}Invalid choice.{RESET}")


def run_chat(character: Character, checkpointer: MongoDBSaver, thread_id: str, diary_executor: ThreadPoolExecutor | None = None):
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
                    },
                    config=graph_config,
                )

            # Extract AI response from result
            ai_message = result["messages"][-1]
            print(f"{GREEN}{BOLD}{character.name}:{RESET} {ai_message.content}")

            # Trigger diary extraction asynchronously after response is shown
            if diary_executor is not None and should_run_diary(result) == "diary":
                state_snapshot = {
                    "messages": list(result.get("messages", [])),
                    "character_name": result.get("character_name", ""),
                    "long_term_facts": list(result.get("long_term_facts", [])),
                    "turn_count": result.get("turn_count", 0),
                }
                diary_executor.submit(_run_diary_safe, state_snapshot)

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

    # Check if Ollama is reachable
    print(f"{DIM}Connecting to Ollama at {config.ollama_base_url}...{RESET}")
    try:
        resp = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            print(f"{GREEN}✓ Ollama connected. Available models: {', '.join(models)}{RESET}")
        else:
            print(f"{YELLOW}⚠ Ollama connected but no models found. Run: ollama pull {config.ollama_model}{RESET}")
            sys.exit(1)

        if not any(config.ollama_model in m for m in models):
            print(f"{YELLOW}⚠ Model '{config.ollama_model}' not found. Run: ollama pull {config.ollama_model}{RESET}")
            sys.exit(1)
    except httpx.ConnectError:
        print(f"{RED}✗ Cannot connect to Ollama. Make sure it's running:{RESET}")
        print(f"  {DIM}ollama serve{RESET}")
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"{RED}✗ Ollama connection timed out.{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}✗ Unexpected error connecting to Ollama: {e}{RESET}")
        sys.exit(1)

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

    print()

    # Thread pool for background diary tasks
    diary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diary")

    def _shutdown_diary():
        print(f"\n{DIM}Waiting for background tasks...{RESET}")
        diary_executor.shutdown(wait=True)

    atexit.register(_shutdown_diary)

    # Main loop: select character → chat → repeat
    while True:
        character = select_character()
        print(f"\n{GREEN}✓ Playing as {BOLD}{character.name}{RESET}")

        # Let user resume existing thread or start new
        existing_thread = select_thread(character, mongo_client)
        thread_id = existing_thread or f"{character.name.lower()}-{uuid.uuid4().hex[:8]}"

        result = run_chat(character, checkpointer, thread_id, diary_executor)

        if result == "quit":
            break
        # result == "switch" → loop back to character selection

    print(f"\n{MAGENTA}{BOLD}Thanks for playing! 🎭{RESET}\n")


if __name__ == "__main__":
    main()
