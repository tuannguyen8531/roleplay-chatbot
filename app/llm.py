"""
LLM Factory and Providers.

Supports easy switching between Ollama and Gemini using inheritance and the Factory pattern.
"""

import os
import threading
from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import config

# Thread-local cache: each thread gets its own LLM instances per provider/model/temperature
_llm_local = threading.local()


def _clean_content(content) -> str:
    """Helper to extract clean text from list content (like Gemini thinking blocks)."""
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


class CleanChatOllama(ChatOllama):
    """Subclass of ChatOllama that cleans structured thinking blocks from output."""

    def invoke(self, *args, **kwargs):
        res = super().invoke(*args, **kwargs)
        if hasattr(res, "content"):
            res.content = _clean_content(res.content)
        return res


class CleanChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Subclass of ChatGoogleGenerativeAI that cleans structured thinking blocks from output."""

    def invoke(self, *args, **kwargs):
        res = super().invoke(*args, **kwargs)
        if hasattr(res, "content"):
            res.content = _clean_content(res.content)
        return res


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers."""

    @abstractmethod
    def get_chat_model(self, model_name: str, temperature: float) -> BaseChatModel:
        """Create and return a LangChain BaseChatModel instance."""
        pass


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM Provider implementation."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_chat_model(self, model_name: str, temperature: float) -> BaseChatModel:
        return CleanChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            num_ctx=config.ollama_num_ctx,
        )


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM Provider implementation."""

    def __init__(self, api_key: str | None = None):
        # Falls back to GOOGLE_API_KEY environment variable if api_key is not passed
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

    def get_chat_model(self, model_name: str, temperature: float) -> BaseChatModel:
        # Check if API key is present
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY in your .env file "
                "or export GOOGLE_API_KEY in your environment."
            )
        return CleanChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature,
        )


class LLMFactory:
    """Factory class to instantiate LLM Providers."""

    @staticmethod
    def get_provider(provider_name: str) -> BaseLLMProvider:
        """Get the concrete LLM Provider instance based on name."""
        name = provider_name.lower().strip()
        if name == "ollama":
            return OllamaLLMProvider(base_url=config.ollama_base_url)
        elif name == "gemini":
            return GeminiLLMProvider(api_key=config.gemini_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> BaseChatModel:
    """
    Get or create a cached LangChain BaseChatModel instance.

    Uses thread-local storage to cache instances based on:
    (provider, model_name, temperature).
    """
    active_provider = provider or config.llm_provider
    active_model = model_name or (
        config.gemini_model if active_provider.lower() == "gemini" else config.ollama_model
    )
    temp = (
        temperature
        if temperature is not None
        else config.ollama_temperature
    )

    # Initialize thread-local storage if not present
    if not hasattr(_llm_local, "llms"):
        _llm_local.llms = {}

    cache_key = (active_provider.lower(), active_model, temp)

    if cache_key not in _llm_local.llms:
        provider_inst = LLMFactory.get_provider(active_provider)
        _llm_local.llms[cache_key] = provider_inst.get_chat_model(active_model, temp)

    return _llm_local.llms[cache_key]
