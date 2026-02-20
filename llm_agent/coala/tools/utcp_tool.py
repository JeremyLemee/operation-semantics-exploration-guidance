from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from llm_agent.coala.tools.coala_tool import CoalaTool

try:
    # Official UTCP Python client (if available).
    from utcp import Client as UTCPClient  # type: ignore
except (
    Exception
):  # pragma: no cover - optional dependency errors are handled gracefully
    UTCPClient = None


class UTCPTool(CoalaTool):
    """
    Adapter to use UTCP (Universal Tool Calling Protocol) tools as CoALA tools.

    The wrapper prefers the official `utcp.Client` when available and falls back to a
    simple HTTP POST call to `${endpoint}/call` with payload:
        {"name": "<tool_name>", "arguments": {...}}
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        *,
        tool_name: Optional[str] = None,
        api_key: Optional[str] = None,
        description: Optional[str] = None,
        client: Any = None,
        default_argument_key: str = "input",
    ):
        super().__init__(name)
        self.endpoint = endpoint.rstrip("/")
        self.tool_name = tool_name or name
        self.api_key = api_key
        self._description = description
        self.default_argument_key = default_argument_key
        # Build UTCP client if the library is installed
        self._client = client
        if self._client is None and UTCPClient is not None:
            try:
                self._client = UTCPClient(self.endpoint, api_key=self.api_key)
            except Exception:
                # Swallow to allow HTTP fallback
                self._client = None

    @property
    def description(self):
        return self._description or ""

    async def _maybe_fetch_description(self) -> None:
        if self._description or not self._client:
            return
        list_fn = getattr(self._client, "list_tools", None)
        if not list_fn:
            return

        tools_result = await self._maybe_await(list_fn)
        if not tools_result:
            return
        # Handle both attribute and dict-like tool metadata
        tools = getattr(tools_result, "tools", tools_result)
        for tool in tools or []:
            t_name = getattr(tool, "name", None) or (
                tool.get("name") if isinstance(tool, dict) else None
            )
            if t_name == self.tool_name:
                self._description = getattr(tool, "description", None) or (
                    tool.get("description") if isinstance(tool, dict) else None
                )
                break

    async def _maybe_await(self, fn, *args, **kwargs):
        """Call sync/async functions uniformly."""
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _call_via_client(self, arguments: Dict[str, Any]):
        if not self._client:
            return None
        call_fn = getattr(self._client, "call_tool", None) or getattr(
            self._client, "call", None
        )
        if not call_fn:
            return None
        return await self._maybe_await(call_fn, self.tool_name, arguments=arguments)

    async def _call_via_http(self, arguments: Dict[str, Any]):
        payload = {"name": self.tool_name, "arguments": arguments}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.endpoint}/call", json=payload, headers=headers
            )
            resp.raise_for_status()
            return resp.json()

    def _normalize_content(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, dict):
            if "content" in result and isinstance(result["content"], str):
                return result["content"]
            if "text" in result and isinstance(result["text"], str):
                return result["text"]
            return str(result)
        text = getattr(result, "content", None)
        if isinstance(text, str):
            return text
        # Try nested content parts if present
        parts = getattr(result, "content", None) or []
        if isinstance(parts, list):
            texts = []
            for part in parts:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    maybe_text = part.get("text")
                    if maybe_text:
                        texts.append(maybe_text)
                else:
                    maybe_text = getattr(part, "text", None)
                    if maybe_text:
                        texts.append(maybe_text)
            if texts:
                return "\n".join(texts)
        return str(result)

    async def ainvoke(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        await self._maybe_fetch_description()

        if isinstance(tool_input, dict):
            arguments = tool_input.get("arguments", tool_input)
        elif isinstance(tool_input, str):
            arguments = {self.default_argument_key: tool_input}
        elif tool_input is None:
            arguments = {}
        else:
            raise TypeError("tool_input must be a dict or string for UTCP tools.")

        result = await self._call_via_client(arguments)
        if result is None:
            result = await self._call_via_http(arguments)

        return {
            "content": self._normalize_content(result),
            "raw": result,
        }
