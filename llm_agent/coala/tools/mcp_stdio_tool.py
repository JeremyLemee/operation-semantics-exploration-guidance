import asyncio
from typing import Optional, Sequence, Mapping

from llm_agent.coala.tools.coala_tool import CoalaTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPStdioTool(CoalaTool):
    def __init__(
        self,
        name: str,
        command: str = "python",
        args: Optional[Sequence[str]] = None,
        *,
        tool_name: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ):
        """
        Example:
            MCPStdioTool(
                name="weather",
                command="python",
                args=["/path/to/server.py"],
                tool_name="get_weather",
            )
        """
        super().__init__(name)
        self.server_params = StdioServerParameters(
            command=command,
            args=list(args or []),
            env=env,
        )
        self.tool_name = tool_name
        self._description = None  # will be filled from server

        # Synchronously run the async init logic to get the description
        asyncio.run(self._fetch_tool_description())

    @property
    def description(self):
        return self._description

    async def _fetch_tool_description(self):
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Bare initialize — some servers reject extra params
                await session.initialize()

                # Fetch all tools
                tools_result = await session.list_tools()

                # Find our tool and store its description
                for tool in tools_result.tools:
                    if tool.name == self.tool_name:
                        self._description = tool.description
                        break

    async def ainvoke(self, tool_input: dict) -> dict:
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                args = tool_input.get("arguments", {})
                result = await session.call_tool(self.tool_name, arguments=args)

                # Normalize content to a simple text string (similar to original)
                text_parts = []
                for part in getattr(result, "content", []) or []:
                    text = getattr(part, "text", None)
                    if text is None and isinstance(part, dict):
                        text = part.get("text")
                    if text:
                        text_parts.append(text)

                return {"content": "\n".join(text_parts) if text_parts else str(result)}
