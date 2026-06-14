import os
from pathlib import Path
from typing import Any

from langchain.chat_models import BaseChatModel

from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama
from pydantic import SecretStr


def _normalize_openai_reasoning(reasoning: Any) -> dict[str, Any] | None:
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        return reasoning
    if isinstance(reasoning, str):
        return {"effort": reasoning}
    if isinstance(reasoning, bool):
        return {"effort": "medium"} if reasoning else None
    raise TypeError(
        "OpenAI reasoning must be None, a bool, a level string, or a dict of reasoning options"
    )


def load_llm(
    provider: str,
    name: str,
    temperature: float | None = None,
    reasoning: bool | str | dict[str, Any] | None = None,
    thinking: bool | str | dict[str, Any] | None = None,
) -> BaseChatModel:
    model_kwargs: dict[str, Any] = {}
    if temperature is not None:
        model_kwargs["temperature"] = temperature

    provider_key = provider.lower()
    if provider_key == "openai":
        openai_reasoning = _normalize_openai_reasoning(
            reasoning if reasoning is not None else thinking
        )
        if openai_reasoning is not None:
            model_kwargs["reasoning"] = openai_reasoning
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model=name, **model_kwargs)
        key_path = Path(__file__).resolve().parent / "API_KEY.txt"
        api_key = None
        if key_path.exists():
            api_key = key_path.read_text().strip() or None
        if api_key:
            return ChatOpenAI(model=name, api_key=SecretStr(api_key), **model_kwargs)
        return ChatOpenAI(model=name, **model_kwargs)
    if provider_key == "ollama":
        ollama_reasoning = reasoning if reasoning is not None else thinking
        if ollama_reasoning is not None:
            model_kwargs["reasoning"] = ollama_reasoning
        return ChatOllama(model=name, **model_kwargs)

    raise ValueError("Only 'openai' and 'ollama' providers are supported")
