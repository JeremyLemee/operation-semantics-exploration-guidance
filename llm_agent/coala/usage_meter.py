# usage_meter.py
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.callbacks import CallbackManager


class UsageTotals:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_names = []  # optional breadcrumb

    def add(self, in_toks: int, out_toks: int, model: Optional[str] = None):
        self.input_tokens += int(in_toks or 0)
        self.output_tokens += int(out_toks or 0)
        if model:
            self.model_names.append(model)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def as_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "models": self.model_names,
        }


def _extract_from_metadata(meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    # Standard LangChain keys
    if "input_tokens" in meta or "output_tokens" in meta:
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    # Some wrappers stash OpenAI-style keys under token_usage
    tu = meta.get("token_usage") if isinstance(meta, dict) else None
    if isinstance(tu, dict):
        return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))
    return None


def extract_usage_from_message(msg: AIMessage) -> Optional[Tuple[int, int]]:
    meta = getattr(msg, "usage_metadata", None)
    if isinstance(meta, dict):
        tup = _extract_from_metadata(meta)
        if tup:
            return tup
    # Some models put usage in response_metadata
    rmeta = getattr(msg, "response_metadata", None)
    if isinstance(rmeta, dict):
        tup = _extract_from_metadata(rmeta)
        if tup:
            return tup
    return None


def extract_model_name(msg: AIMessage) -> Optional[str]:
    rmeta = getattr(msg, "response_metadata", None)
    if isinstance(rmeta, dict):
        # Common spots
        return rmeta.get("model") or rmeta.get("model_name")
    return None


@contextmanager
def usage_meter():
    """Yields (totals, callbacks_config) and auto-accumulates universal counts."""
    totals = UsageTotals()
    uni = TokenCounterCallbackHandler()
    cm = CallbackManager([uni])
    try:
        yield totals, {"callbacks": cm.handlers}
    finally:
        # Pull counts estimated by the universal handler
        in_toks = getattr(uni, "total_prompt_tokens", 0)
        out_toks = getattr(uni, "total_completion_tokens", 0)
        if in_toks or out_toks:
            totals.add(in_toks, out_toks)
