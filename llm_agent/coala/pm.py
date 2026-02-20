from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
from urllib.parse import urlparse, urlunparse

from langchain_core.messages import AIMessage

from llm_agent.coala.tools.coala_tool import CoalaTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


@dataclass
class MCPServerRegistration:
    """Holds connection details for an MCP server."""

    name: str
    transport: str  # "stdio" or "http"
    server_url: Optional[str] = None
    command: Optional[str] = None
    args: Sequence[str] = field(default_factory=list)
    env: Optional[Mapping[str, str]] = None
    known_tools: Set[str] = field(default_factory=set)
    tool_definitions: Dict[str, Any] = field(default_factory=dict)

    @asynccontextmanager
    async def session(self):
        if self.transport == "http":
            if not self.server_url:
                raise ValueError("server_url is required for HTTP MCP servers")
            async with streamable_http_client(self.server_url) as (
                read_stream,
                write_stream,
                _sid,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
        else:
            params = StdioServerParameters(
                command=self.command or "python",
                args=list(self.args),
                env=self.env,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session


class MCPRegisteredTool(CoalaTool):
    """Tool wrapper that calls back into an MCP server."""

    def __init__(
        self,
        name: str,
        description: str,
        server: MCPServerRegistration,
        required_fields: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self._description = description or ""
        self.server = server
        self.required_fields = required_fields or []

    @property
    def description(self):
        return self._description

    async def ainvoke(self, tool_input: Any) -> Dict[str, Any]:
        if isinstance(tool_input, dict):
            arguments = tool_input.get("arguments", tool_input)
        elif isinstance(tool_input, str) and len(self.required_fields) == 1:
            arguments = {self.required_fields[0]: tool_input}
        elif tool_input is None:
            arguments = {}
        else:
            raise TypeError(
                "tool_input must be a dict (or a string if the tool has one required argument)."
            )

        async with self.server.session() as session:
            result = await session.call_tool(self.name, arguments=arguments)
            parts = []
            for c in getattr(result, "content", None) or []:
                text = getattr(c, "text", None)
                if text is None and isinstance(c, dict):
                    text = c.get("text")
                parts.append(text if text is not None else str(c))
            return {"content": "\n".join(parts)}


class ProceduralMemory:
    def __init__(self, llm=None, max_tools=10, max_token_limit=1000):
        self.tools: Dict[str, CoalaTool] = {}
        self.llm = llm
        self.max_tools = max_tools
        self.max_token_limit = max_token_limit
        self.mcp_servers: Dict[str, MCPServerRegistration] = {}
        self.tool_providers: Dict[str, Set[str]] = {}
        self.tool_owner: Dict[str, str] = {}
        self.plans: Dict[str, Any] = {}

    def get_tool(self, name):
        return self.tools.get(name)

    def add_tool(self, tool: CoalaTool):
        self.tools[tool.name] = tool
        self.tool_providers.setdefault(tool.name, set()).add("manual")
        self.tool_owner[tool.name] = "manual"

    def register_mcp_server(
        self,
        name: str,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[Sequence[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ):
        """
        Register an MCP server so its tools can be synchronized into procedural memory.

        Provide either server_url (for HTTP/streamable MCP servers) or command (+args/env) for stdio servers.
        """
        if server_url and command:
            raise ValueError(
                "Provide either server_url or command/args for MCP server registration, not both."
            )
        if not server_url and not command:
            raise ValueError(
                "MCP server registration requires a server_url or a command to run."
            )

        if server_url:
            server_url = self._normalize_mcp_url(server_url)
        transport = "http" if server_url else "stdio"
        registration = MCPServerRegistration(
            name=name,
            transport=transport,
            server_url=server_url,
            command=command,
            args=list(args or []),
            env=env,
        )
        self.mcp_servers[name] = registration

    @staticmethod
    def _normalize_mcp_url(server_url: str) -> str:
        parsed = urlparse(server_url)
        path = parsed.path or ""
        if not path or path == "/":
            path = "/mcp"
        elif not path.rstrip("/").endswith("/mcp"):
            path = path.rstrip("/") + "/mcp"
        return urlunparse(parsed._replace(path=path))

    def _build_tool_from_provider(
        self, tool_name: str, provider: str
    ) -> Optional[CoalaTool]:
        if provider == "manual":
            return self.tools.get(tool_name)
        server = self.mcp_servers.get(provider)
        if not server:
            return None
        tool_def = server.tool_definitions.get(tool_name)
        if tool_def is None:
            return None
        input_schema = getattr(tool_def, "input_schema", None) or getattr(
            tool_def, "inputSchema", None
        )
        required = (
            list(input_schema.get("required", []))
            if isinstance(input_schema, dict)
            else []
        )
        return MCPRegisteredTool(
            tool_name,
            getattr(tool_def, "description", "") or "",
            server,
            required_fields=required,
        )

    async def sync_mcp_tools(self):
        """Synchronize the tool list with all registered MCP servers."""
        for server in self.mcp_servers.values():
            try:
                async with server.session() as session:
                    tools_result = await session.list_tools()
            except Exception as exc:
                print(f"Failed to sync tools from MCP server '{server.name}': {exc}")
                continue

            seen: Set[str] = set()
            for tool_def in getattr(tools_result, "tools", []) or []:
                tool_name = tool_def.name
                seen.add(tool_name)
                server.tool_definitions[tool_name] = tool_def
                providers = self.tool_providers.setdefault(tool_name, set())
                providers.add(server.name)

                if self.tool_owner.get(tool_name) == "manual":
                    continue

                input_schema = getattr(tool_def, "input_schema", None) or getattr(
                    tool_def, "inputSchema", None
                )
                required = (
                    list(input_schema.get("required", []))
                    if isinstance(input_schema, dict)
                    else []
                )
                self.tools[tool_name] = MCPRegisteredTool(
                    tool_name,
                    getattr(tool_def, "description", "") or "",
                    server,
                    required_fields=required,
                )
                self.tool_owner[tool_name] = server.name

            removed_tools = server.known_tools - seen
            for tool_name in removed_tools:
                server.tool_definitions.pop(tool_name, None)
                providers = self.tool_providers.get(tool_name, set())
                providers.discard(server.name)

                if not providers:
                    self.tools.pop(tool_name, None)
                    self.tool_owner.pop(tool_name, None)
                elif self.tool_owner.get(tool_name) == server.name:
                    next_owner = (
                        "manual" if "manual" in providers else next(iter(providers))
                    )
                    replacement = self._build_tool_from_provider(tool_name, next_owner)
                    if replacement:
                        self.tools[tool_name] = replacement
                        self.tool_owner[tool_name] = next_owner
            server.known_tools = seen

    @property
    def memory_variables(self) -> List[str]:
        return list(self.tools.keys())

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        d = {}
        for key, value in inputs.items():
            for k, t in self.tools.items():
                if key == t.name or key == t.description:
                    d[k] = t
                elif self.llm is not None:
                    print("TODO")
            for k2, p in self.plans.items():
                if key == k2:  # TODO: update
                    d[k2] = p

        return d

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        for key, value in inputs.items():
            if isinstance(value, CoalaTool):
                self.tools[value.name] = value

    def clear(self) -> None:
        self.tools = {}
        self.tool_providers = {}
        self.tool_owner = {}

    def get_tool_descriptions(self):
        return [f"{tool.name}: {tool.description}" for tool in self.tools.values()]

    def retrieve_tools(self, query=None):
        if not query:
            return list(self.tools.values())[: self.max_tools]

        tool_descriptions = self.get_tool_descriptions()
        retrieval_prompt = (
            "You are selecting tools for a language agent."
            " Based on the following query, return the most relevant tools:"
            f"\n\nQuery: {query}\n"
            f"\nAvailable tools:\n{chr(10).join(tool_descriptions)}"
            "\n\nList the top relevant tools by name only, separated by commas."
        )
        response = self.llm.invoke(retrieval_prompt)
        response_str = ""
        if isinstance(response, str):
            response_str = response
        elif isinstance(response, AIMessage):
            response_str = response.text()
        selected_tool_names = [name.strip() for name in response_str.split(",")]
        return [
            tool for tool in self.tools.values() if tool.name in selected_tool_names
        ][: self.max_tools]
