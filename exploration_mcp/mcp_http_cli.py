#!/usr/bin/env python3
"""
mcp_http_cli.py — tiny MCP Streamable HTTP client (Python SDK)

Usage:
  python mcp_http_cli.py https://example.com/mcp

Optional headers (repeatable):
  python mcp_http_cli.py https://example.com/mcp -H "Authorization: Bearer TOKEN" -H "X-Trace-Id: 123"

Requires:
  pip install "mcp[cli]" httpx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from typing import Any, Dict, Optional, Tuple

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client


def _parse_json_args(s: str) -> Dict[str, Any]:
    s = s.strip()
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError('tool arguments must be a JSON object (e.g. {"a":1})')
    return obj


def _schema_props(schema: Optional[dict]) -> Tuple[dict, set, dict]:
    """Return (properties, required_set, schema_root) for a JSON schema-ish dict."""
    if not isinstance(schema, dict):
        return {}, set(), {}
    props = (
        schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    )
    req = schema.get("required") if isinstance(schema.get("required"), list) else []
    return props, set(x for x in req if isinstance(x, str)), schema


def _schema_type(s: dict) -> str:
    t = s.get("type")
    if isinstance(t, str):
        return t
    if isinstance(t, list) and t:
        return "|".join(str(x) for x in t)
    # Fallbacks for schema variants
    if "enum" in s:
        return "enum"
    if "oneOf" in s:
        return "oneOf"
    if "anyOf" in s:
        return "anyOf"
    return "unknown"


def _format_param_line(name: str, s: dict, required: bool) -> str:
    t = _schema_type(s)
    desc = (s.get("description") or "").strip()
    default = s.get("default", None)
    enum = s.get("enum", None)

    bits = [name, f"({t}{', required' if required else ''})"]
    if default is not None:
        bits.append(f"default={default!r}")
    if isinstance(enum, list) and enum:
        # keep it short-ish
        preview = enum if len(enum) <= 8 else enum[:8] + ["…"]
        bits.append(f"enum={preview!r}")

    line = "  - " + " ".join(bits)
    if desc:
        line += f"\n      {desc}"
    return line


def _print_tool_list(resp: types.ListToolsResult) -> None:
    if not resp.tools:
        print("(no tools)")
        return

    for t in resp.tools:
        desc = (t.description or "").strip()
        print(f"- {t.name}")
        if desc:
            print(f"    {desc}")

        # NEW: print parameters from inputSchema (JSON Schema)
        schema = getattr(t, "inputSchema", None)
        props, req_set, _ = _schema_props(schema)

        if not props:
            print("    params: (none)")
            continue

        print("    params:")
        for pname, pschema in props.items():
            if not isinstance(pschema, dict):
                pschema = {}
            print(_format_param_line(pname, pschema, pname in req_set))


def _tool_to_dict(tool: types.Tool) -> dict:
    try:
        data = tool.model_dump()
    except Exception:
        try:
            data = tool.model_dump(mode="json")
        except Exception:
            data = {
                "name": getattr(tool, "name", None),
                "description": getattr(tool, "description", None),
                "inputSchema": getattr(tool, "inputSchema", None),
            }
    return data


def _print_resource_list(resp: types.ListResourcesResult) -> None:
    if not resp.resources:
        print("(no resources)")
        return
    for r in resp.resources:
        print(f"- {r.uri}")
        print(f"    name: {r.name}")
        if r.mimeType:
            print(f"    mime: {r.mimeType}")
        if r.description:
            print(f"    desc: {r.description}")


def _print_read_result(resp: types.ReadResourceResult) -> None:
    if not resp.contents:
        print("(empty)")
        return

    for i, c in enumerate(resp.contents, start=1):
        if isinstance(c, types.TextResourceContents):
            print(f"--- content[{i}] (text) ---")
            print(c.text)
            continue
        if isinstance(c, types.BlobResourceContents):
            print(f"--- content[{i}] (blob) ---")
            blob_data = c.blob or ""
            print(f"(blob mime={c.mimeType!r}, len={len(blob_data)} chars)")
            continue

        content_type = getattr(c, "type", "unknown")
        print(f"--- content[{i}] ({content_type}) ---")
        try:
            print(c.model_dump_json(indent=2))
        except Exception:
            print(repr(c))


def _print_call_result(resp: types.CallToolResult) -> None:
    if resp.content:
        for i, c in enumerate(resp.content, start=1):
            print(f"--- content[{i}] ({c.type}) ---")
            if isinstance(c, types.TextContent):
                print(c.text)
            elif isinstance(c, types.BlobResourceContents):
                data = c.data or ""
                print(f"(blob mime={c.mimeType!r}, len={len(data)} chars)")
            else:
                try:
                    print(c.model_dump_json(indent=2))
                except Exception:
                    print(repr(c))


HELP = """Commands:
  help
  session
  tools [cursor]
  resources [cursor]
  read <uri>
  desc <tool_name>
  call [-p] <tool_name> [jsonArgs]
      -p  prompt for parameters interactively based on tool inputSchema
  quit | exit

Examples:
  tools
  resources
  read greeting://World
  call add {"a":2,"b":5}
  call -p add
  desc add
"""


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def _find_tool(session: ClientSession, tool_name: str) -> Optional[types.Tool]:
    """Paginate tools/list until the tool is found (or exhausted)."""
    cursor: Optional[str] = None
    while True:
        resp = await session.list_tools(cursor=cursor)
        for t in resp.tools or []:
            if t.name == tool_name:
                return t
        cursor = resp.nextCursor
        if not cursor:
            return None


async def _read_resource_raw(
    session: ClientSession, uri_str: str
) -> types.ReadResourceResult:
    """
    Read a resource without AnyUrl normalization to avoid altering custom URIs.
    """
    params = types.ReadResourceRequestParams.model_construct(uri=uri_str)
    request = types.ReadResourceRequest(params=params)
    return await session.send_request(
        types.ClientRequest(request), types.ReadResourceResult
    )


def _coerce_from_schema(user_text: str, schema: dict) -> Any:
    """
    Coerce a single user-entered value to something reasonable based on JSON Schema.
    For complex types (object/array), expects JSON.
    """
    user_text = user_text.strip()
    t = _schema_type(schema)
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        # Accept as JSON or string match
        try:
            v = json.loads(user_text)
        except Exception:
            v = user_text
        if v not in enum:
            raise ValueError(f"value must be one of {enum!r}")
        return v

    if t in ("object", "array"):
        return json.loads(user_text)

    if t == "boolean":
        lowered = user_text.lower()
        if lowered in ("true", "t", "yes", "y", "1"):
            return True
        if lowered in ("false", "f", "no", "n", "0"):
            return False
        raise ValueError("expected a boolean (true/false, yes/no, 1/0)")

    if t == "integer":
        return int(user_text)

    if t == "number":
        return float(user_text)

    # string / unknown / union: accept raw string, but allow JSON literals if user types them
    try:
        return json.loads(user_text)
    except Exception:
        return user_text


async def _prompt_for_arguments(tool: types.Tool) -> Dict[str, Any]:
    """Ask the user for each parameter defined in tool.inputSchema, one by one."""
    schema = getattr(tool, "inputSchema", None)
    props, req_set, _ = _schema_props(schema)

    if not props:
        print("(tool has no declared params; calling with {})")
        return {}

    args: Dict[str, Any] = {}
    print(f"Prompting for parameters of tool: {tool.name}")
    print("(Press Enter to skip optional params; JSON required for object/array.)")

    for pname, pschema in props.items():
        if not isinstance(pschema, dict):
            pschema = {}

        required = pname in req_set
        desc = (pschema.get("description") or "").strip()
        default = pschema.get("default", None)
        enum = pschema.get("enum", None)

        hint_bits = [f"type={_schema_type(pschema)}"]
        if required:
            hint_bits.append("required")
        if default is not None:
            hint_bits.append(f"default={default!r}")
        if isinstance(enum, list) and enum:
            preview = enum if len(enum) <= 8 else enum[:8] + ["…"]
            hint_bits.append(f"enum={preview!r}")

        print(f"\n{pname} ({', '.join(hint_bits)})")
        if desc:
            print(f"  {desc}")

        while True:
            raw = await ainput(f"  value for {pname}: ")
            if raw.strip() == "":
                if default is not None:
                    args[pname] = default
                    break
                if required:
                    print("  This parameter is required.")
                    continue
                # optional + blank => omit
                break

            try:
                args[pname] = _coerce_from_schema(raw, pschema)
                break
            except json.JSONDecodeError:
                print(
                    '  Invalid JSON. For object/array, enter valid JSON (e.g. {"k":1} or [1,2]).'
                )
            except Exception as e:
                print(f"  {e}")

    return args


async def repl(session: ClientSession) -> None:
    print(HELP)
    while True:
        line = (await ainput("mcp> ")).strip()
        if not line:
            continue

        parts = shlex.split(line)
        cmd = parts[0].lower()
        rest = parts[1:]

        try:
            if cmd in ("quit", "exit"):
                return

            if cmd == "help":
                print(HELP)
                continue

            if cmd == "session":
                print("(session info is transport-specific; see startup banner)")
                continue

            if cmd == "tools":
                cursor = rest[0] if rest else None
                resp = await session.list_tools(cursor=cursor)
                _print_tool_list(resp)
                if resp.nextCursor:
                    print(f"nextCursor: {resp.nextCursor}")
                continue

            if cmd == "resources":
                cursor = rest[0] if rest else None
                resp = await session.list_resources(cursor=cursor)
                _print_resource_list(resp)
                if resp.nextCursor:
                    print(f"nextCursor: {resp.nextCursor}")
                continue

            if cmd == "read":
                if not rest:
                    print("Usage: read <uri>")
                    continue
                uri_str = rest[0]
                resp = await _read_resource_raw(session, uri_str)
                _print_read_result(resp)
                continue

            if cmd == "desc":
                if not rest:
                    print("Usage: desc <tool_name>")
                    continue
                tool_name = rest[0]
                tool = await _find_tool(session, tool_name)
                if tool is None:
                    print(f"Tool not found: {tool_name}")
                    continue
                print(json.dumps(_tool_to_dict(tool), indent=2))
                continue

            if cmd == "call":
                if not rest:
                    print("Usage: call [-p] <tool_name> [jsonArgs]")
                    continue

                # NEW: -p option (prompt for params)
                prompt_mode = False
                if rest[0] == "-p":
                    prompt_mode = True
                    rest = rest[1:]

                if not rest:
                    print("Usage: call [-p] <tool_name> [jsonArgs]")
                    continue

                tool_name = rest[0]
                json_args = " ".join(rest[1:]) if len(rest) > 1 else ""

                if prompt_mode:
                    tool = await _find_tool(session, tool_name)
                    if tool is None:
                        print(f"Tool not found: {tool_name}")
                        continue
                    args = await _prompt_for_arguments(tool)
                else:
                    args = _parse_json_args(json_args)

                resp = await session.call_tool(tool_name, arguments=args)
                _print_call_result(resp)
                continue

            print(f"Unknown command: {cmd} (type 'help')")

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="MCP Streamable HTTP CLI (Python SDK)")
    parser.add_argument(
        "url", help="MCP Streamable HTTP endpoint URL, e.g. https://host/mcp"
    )
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        default=[],
        help='Extra HTTP header, repeatable (e.g. -H "Authorization: Bearer TOKEN")',
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout seconds"
    )
    args = parser.parse_args()

    headers: Dict[str, str] = {}
    for h in args.header:
        if ":" not in h:
            raise SystemExit(f"Bad header {h!r}; expected 'Name: value'")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()

    async with httpx.AsyncClient(headers=headers, timeout=args.timeout) as http_client:
        async with streamable_http_client(url=args.url, http_client=http_client) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                sid = get_session_id() if callable(get_session_id) else None
                if sid:
                    print(f"Connected. Mcp-Session-Id: {sid}")
                else:
                    print(
                        "Connected. (No Mcp-Session-Id exposed; server may be stateless.)"
                    )

                await repl(session)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
