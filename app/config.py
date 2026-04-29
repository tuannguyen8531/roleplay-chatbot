"""
Configuration for the AI Roleplay Chatbot.
Loads settings from .env file or environment variables.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env file into os.environ before reading config
# interpolate=True enables ${VAR} variable expansion within .env
load_dotenv(interpolate=True)


@dataclass
class Config:
    """Application configuration."""

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_utility_model: str = ""  # Smaller model for summary/diary (empty = use main model)
    ollama_temperature: float = 0.8
    ollama_num_ctx: int = 8192  # Context window size

    @property
    def utility_model(self) -> str:
        """Model used for summarization and diary (falls back to main model)."""
        return self.ollama_utility_model or self.ollama_model

    # MongoDB settings
    mongodb_user: str = "chatbot"
    mongodb_password: str = "chatbot"
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_db_name: str = "chatbot"

    @property
    def mongodb_uri(self) -> str:
        """Build MongoDB URI from individual components."""
        return f"mongodb://{self.mongodb_user}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}"

    # Chat settings
    max_history_messages: int = 20  # Keep last N messages before trimming
    diary_interval: int = 5  # Run diary extraction every N turns
    streaming: bool = True  # Stream responses token-by-token

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_utility_model=os.getenv("OLLAMA_UTILITY_MODEL", cls.ollama_utility_model),
            ollama_temperature=float(
                os.getenv("OLLAMA_TEMPERATURE", str(cls.ollama_temperature))
            ),
            ollama_num_ctx=int(
                os.getenv("OLLAMA_NUM_CTX", str(cls.ollama_num_ctx))
            ),
            mongodb_user=os.getenv("MONGODB_USER", cls.mongodb_user),
            mongodb_password=os.getenv("MONGODB_PASSWORD", cls.mongodb_password),
            mongodb_host=os.getenv("MONGODB_HOST", cls.mongodb_host),
            mongodb_port=int(os.getenv("MONGODB_PORT", str(cls.mongodb_port))),
            mongodb_db_name=os.getenv("MONGODB_DB_NAME", cls.mongodb_db_name),
            max_history_messages=int(
                os.getenv("MAX_HISTORY_MESSAGES", str(cls.max_history_messages))
            ),
            diary_interval=int(
                os.getenv("DIARY_INTERVAL", str(cls.diary_interval))
            ),
            streaming=os.getenv("STREAMING", "true").lower() == "true",
        )


# Global config instance
config = Config.from_env()
