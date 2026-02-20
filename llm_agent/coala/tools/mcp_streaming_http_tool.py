import asyncio
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPStreamingHTTPTool:
    def __init__(
        self,
        name: str,
        server_url: str = "http://localhost:8000/mcp/",
        tool_name: Optional[str] = None,
    ):
        self.name = name
        self.server_url = server_url.rstrip("/")  # just in case
        self.tool_name = tool_name or name
        self._description: Optional[str] = None
        self._input_schema: Optional[Dict[str, Any]] = None
        self._required: list[str] = []
        self._param_descriptions: list[tuple[str, str, bool]] = []
        # Fetch metadata now (or schedule if we're already in an event loop)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._fetch_tool_meta())
        else:
            loop.create_task(self._fetch_tool_meta())

    @property
    def description(self) -> Optional[str]:
        return self._description

    def describe(self) -> str:
        base_desc = self._description or ""
        if not self._param_descriptions:
            return base_desc
        parts = []
        for name, desc, required in self._param_descriptions:
            required_tag = " (required)" if required else ""
            detail = f"{name}{required_tag}"
            if desc:
                detail = f"{detail} - {desc}"
            parts.append(detail)
        params_text = "; ".join(parts)
        if base_desc:
            return f"{base_desc} | params: {params_text}"
        return f"params: {params_text}"

    async def _fetch_tool_meta(self) -> None:
        async with streamable_http_client(self.server_url) as (
            read_stream,
            write_stream,
            _sid,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                for t in tools.tools:
                    if t.name == self.tool_name:
                        self._description = t.description
                        self._input_schema = getattr(
                            t, "input_schema", None
                        ) or getattr(t, "inputSchema", None)
                        if isinstance(self._input_schema, dict):
                            self._required = list(
                                self._input_schema.get("required", [])
                            )
                            self._param_descriptions = self._extract_param_descriptions(
                                self._input_schema
                            )
                        break

    def _extract_param_descriptions(
        self, schema: Dict[str, Any]
    ) -> list[tuple[str, str, bool]]:
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = (
            set(schema.get("required", [])) if isinstance(schema, dict) else set()
        )
        params: list[tuple[str, str, bool]] = []
        if isinstance(properties, dict):
            for name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    params.append((name, "", name in required))
                    continue
                desc = (
                    prop_schema.get("description", "")
                    or prop_schema.get("title", "")
                    or ""
                )
                params.append((name, desc, name in required))
        return params

    async def ainvoke(self, tool_input: Any) -> Dict[str, Any]:
        """
        Call the MCP tool. Accepts either:
          - a flat dict of args (recommended): {"s": "..."}
          - a convenience string *iff* the tool has exactly one required field
          - a nested shape {"arguments": {...}} for backward-compat
        """
        # Normalize arguments
        if (
            isinstance(tool_input, dict)
            and "arguments" in tool_input
            and isinstance(tool_input["arguments"], dict)
        ):
            args = tool_input["arguments"]
        elif isinstance(tool_input, dict):
            args = tool_input
        elif isinstance(tool_input, str) and len(self._required) == 1:
            # Let callers pass a bare string for single-arg tools
            args = {self._required[0]: tool_input}
        else:
            raise TypeError(
                "tool_input must be a dict of arguments (e.g., {'s': '...'})."
            )

        async with streamablehttp_client(self.server_url) as (
            read_stream,
            write_stream,
            _sid,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(self.tool_name, arguments=args)
                # Extract text content parts
                parts = []
                for c in result.content or []:
                    if getattr(c, "type", None) == "text" and hasattr(c, "text"):
                        parts.append(c.text)
                    else:
                        parts.append(str(c))
                return {"content": "\n".join(parts), "raw": result}
