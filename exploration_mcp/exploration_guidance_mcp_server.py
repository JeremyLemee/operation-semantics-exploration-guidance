import json
import re
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from mcp.server.fastmcp import FastMCP
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS

import requests

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import ExplorationOntology  # noqa: E402
import HTTPOntology  # noqa: E402


mcp = FastMCP(name="Exploration Guidance", host="0.0.0.0", port=8100)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8101

DESCRIPTION_MODES = {
    "full_guidance",
    "no_guidance",
    "outcome",
    "danger",
    "explorability",
    "outcome_only",
}
MODE_ALIASES = {
    "all": "full_guidance",
    "full": "full_guidance",
    "full_guidance": "full_guidance",
    "full guidance": "full_guidance",
    "none": "no_guidance",
    "no guidance": "no_guidance",
    "no_guidance": "no_guidance",
    "outcome": "outcome",
    "danger": "danger",
    "explorability": "explorability",
    "outcome only": "outcome_only",
    "outcome_only": "outcome_only",
}

TOOL_DESCRIPTION_MODE = "full_guidance"
LAST_GUIDANCE_GRAPH: Graph | None = None
DYNAMIC_TOOL_NAMES: set[str] = set()

_flask_app = Flask("exploration_guidance_state")


@dataclass
class GuidedOperation:
    name: str
    description: str
    method: str
    url: str
    body: str | None = None


@mcp.tool()
def http_request(url: str) -> str:
    """Perform a GET HTTP request and load exploration guidance from its Link header."""
    response = requests.get(url)
    _apply_guidance_from_response(response)
    return response.text


@mcp.tool()
def http_post_request(url: str, body: str) -> str:
    """Perform a POST HTTP request with a JSON body and load exploration guidance.

    The body is a JSON object encoded as text, for example {"name": "bob"}.
    Escaped JSON strings are accepted for compatibility with interactive clients.
    """
    response = requests.post(url, json=_parse_json_body(body))
    _apply_guidance_from_response(response)
    return response.text


def _parse_json_body(body: str) -> Any:
    parsed = json.loads(body)
    if isinstance(parsed, str):
        try:
            return json.loads(parsed)
        except json.JSONDecodeError:
            return parsed
    return parsed


def _apply_guidance_from_response(response: requests.Response) -> None:
    guidance_url = _guidance_link(response)
    if guidance_url is None:
        return

    graph = Graph()
    graph.parse(guidance_url)
    _apply_guidance(graph)


def _guidance_link(response: requests.Response) -> str | None:
    link_header = response.headers.get("Link")
    if not link_header:
        return None

    pattern = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')
    for part in (item.strip() for item in link_header.split(",")):
        match = pattern.match(part)
        if match and match.group(2) == "guidance":
            return match.group(1)
    return None


def _apply_guidance(graph: Graph) -> None:
    global LAST_GUIDANCE_GRAPH

    LAST_GUIDANCE_GRAPH = graph
    _clear_dynamic_tools()

    for operation in _extract_operations(graph):
        mcp.add_tool(
            _operation_tool(operation),
            name=operation.name,
            description=operation.description,
        )
        DYNAMIC_TOOL_NAMES.add(operation.name)


def _clear_dynamic_tools() -> None:
    for name in list(DYNAMIC_TOOL_NAMES):
        try:
            mcp.remove_tool(name)
        except Exception:
            pass
    DYNAMIC_TOOL_NAMES.clear()


def _extract_operations(graph: Graph) -> list[GuidedOperation]:
    operations = []
    existing_names = _current_tool_names()

    for node in _operation_nodes(graph):
        url_value = graph.value(node, HTTPOntology.requestURI)
        if url_value is None:
            continue

        name = _unique_name(_node_name(graph, node), existing_names)
        existing_names.add(name)

        method_value = graph.value(node, HTTPOntology.methodName)
        body_value = graph.value(node, HTTPOntology.body)

        operations.append(
            GuidedOperation(
                name=name,
                description=_operation_description(graph, node),
                method=str(method_value) if method_value is not None else "GET",
                url=str(url_value),
                body=str(body_value) if body_value is not None else None,
            )
        )

    return operations


def _operation_nodes(graph: Graph) -> set:
    nodes = set()
    for operation_type in (
        HTTPOntology.Request,
        ExplorationOntology.Operation,
        ExplorationOntology.ExplorableOperation,
        ExplorationOntology.NonExplorableOperation,
    ):
        nodes.update(graph.subjects(RDF.type, operation_type))
    return nodes


def _current_tool_names() -> set[str]:
    names = set()
    for attr in ("tools", "_tools"):
        tool_obj = getattr(mcp, attr, None)
        if isinstance(tool_obj, dict):
            names.update(str(name) for name in tool_obj)
    tool_manager = getattr(mcp, "_tool_manager", None)
    managed_tools = getattr(tool_manager, "_tools", None)
    if isinstance(managed_tools, dict):
        names.update(str(name) for name in managed_tools)
    return names


def _node_name(graph: Graph, node) -> str:
    label = graph.value(node, RDFS.label)
    if isinstance(label, Literal) and str(label).strip():
        return str(label).strip()

    if isinstance(node, URIRef):
        uri = str(node).rstrip("/#")
        if "#" in uri:
            return uri.rsplit("#", 1)[-1]
        if "/" in uri:
            return uri.rsplit("/", 1)[-1]

    return f"operation_{uuid.uuid4().hex}"


def _unique_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        return name

    index = 2
    while f"{name}{index}" in existing:
        index += 1
    return f"{name}{index}"


def _operation_description(graph: Graph, node) -> str:
    mode = TOOL_DESCRIPTION_MODE

    if mode == "no_guidance":
        return "This is an operation."

    parts = []
    operation_comment = _comment(graph, node)
    if mode == "full_guidance" and operation_comment:
        parts.append(operation_comment)

    if mode in {"full_guidance", "outcome", "danger", "explorability"}:
        explorability = _explorability_description(graph, node)
        if explorability:
            parts.append(explorability)

    if mode in {"full_guidance", "outcome", "outcome_only"}:
        outcome = _outcome_description(graph, node)
        if outcome:
            parts.append(outcome)

    if mode in {"full_guidance", "danger"}:
        danger = _danger_description(graph, node)
        if danger:
            parts.append(danger)

    return " ".join(parts) if parts else "This is an operation."


def _explorability_description(graph: Graph, node) -> str | None:
    if (node, RDF.type, ExplorationOntology.ExplorableOperation) in graph:
        return "Operation is explorable."
    if (node, RDF.type, ExplorationOntology.NonExplorableOperation) in graph:
        return "Operation is not explorable."
    return None


def _outcome_description(graph: Graph, node) -> str | None:
    comments = [
        comment
        for outcome in graph.objects(node, ExplorationOntology.hasOutcome)
        if (comment := _comment(graph, outcome))
    ]
    if not comments:
        return None
    label = "Outcome" if len(comments) == 1 else "Outcomes"
    return f"{label}: {'; '.join(comments)}"


def _danger_description(graph: Graph, node) -> str | None:
    comments = [
        comment
        for cause in graph.objects(node, ExplorationOntology.hasDangerCause)
        if (comment := _comment(graph, cause))
    ]
    if not comments:
        return None
    return f"Danger causes: {'; '.join(comments)}"


def _comment(graph: Graph, node) -> str | None:
    value = graph.value(node, RDFS.comment)
    if isinstance(value, Literal):
        text = str(value).strip()
        return text or None
    return None


def _operation_tool(operation: GuidedOperation):
    def tool():
        response = _perform_operation(operation)
        _apply_guidance_from_response(response)
        return _response_payload(response)

    return tool


def _perform_operation(operation: GuidedOperation) -> requests.Response:
    request_kwargs = {}
    if operation.body is not None:
        try:
            request_kwargs["json"] = json.loads(operation.body)
        except json.JSONDecodeError:
            request_kwargs["data"] = operation.body

    return requests.request(operation.method.upper(), operation.url, **request_kwargs)


def _response_payload(response: requests.Response) -> dict:
    payload = {"status_code": response.status_code, "ok": response.ok}
    content_type = response.headers.get("Content-Type", "").lower()

    if "application/json" in content_type:
        try:
            payload["content"] = response.json()
            return payload
        except ValueError:
            pass

    payload["content"] = response.text
    return payload


def _normalize_mode(mode: str) -> str | None:
    key = mode.strip().lower().replace("-", "_")
    return MODE_ALIASES.get(key)


def _set_tool_description_mode(mode: str) -> tuple[bool, str]:
    global TOOL_DESCRIPTION_MODE

    normalized = _normalize_mode(mode)
    if normalized is None:
        available = ", ".join(sorted(DESCRIPTION_MODES))
        return False, f"Unknown tool_description_mode '{mode}'. Available: {available}."

    TOOL_DESCRIPTION_MODE = normalized
    if LAST_GUIDANCE_GRAPH is not None:
        _apply_guidance(LAST_GUIDANCE_GRAPH)
    return True, normalized


def _serialize_state() -> dict:
    return {
        "tool_description_mode": TOOL_DESCRIPTION_MODE,
        "tool_description_modes": sorted(DESCRIPTION_MODES),
        "has_guidance_graph": LAST_GUIDANCE_GRAPH is not None,
    }


@_flask_app.get("/state")
def get_state():
    return jsonify(_serialize_state())


@_flask_app.patch("/state")
def update_state():
    payload = request.get_json(silent=True) or {}

    if "tool_description_mode" in payload:
        ok, result = _set_tool_description_mode(str(payload["tool_description_mode"]))
        if not ok:
            return jsonify({"error": result}), 400

    return jsonify(_serialize_state())


def _run_flask_server() -> None:
    _flask_app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)


def _start_flask_thread() -> None:
    thread = threading.Thread(
        target=_run_flask_server,
        name="guidance-flask",
        daemon=True,
    )
    thread.start()


if __name__ == "__main__":
    _start_flask_thread()
    mcp.run(transport="streamable-http")
