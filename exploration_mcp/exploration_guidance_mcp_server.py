import uuid
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
import threading

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from mcp.server.fastmcp import FastMCP
from flask import Flask, jsonify, request

from rdflib import Graph, URIRef, RDF, Literal, Namespace
from rdflib.namespace import RDFS

import requests

import re

import json

import ExplorationOntology
import HTTPOntology
from exploration_mcp.operation import Operation

mcp = FastMCP(name="Exploration Guidance", host="0.0.0.0", port=8100)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8101
MAZE_HTTP_BASE = "http://localhost:5001"

# Designer-configurable defaults.
DEFAULT_POLICY = "all"  # all, explorable, explorable_exits no_fee, no_trap_final, explorable_or_safe_non_explorable, explorable_or_owned_or_affordable, explorable_exits
SHOW_SET_POLICY_TOOL = False
SHOW_SET_BUDGET_TOOL = False
# Controls which details appear in operation tool descriptions.
# Modes: "all", "outcome", "none", "explorability", "no outcome".
TOOL_DESCRIPTION_MODE = "all"

POLICIES = {}
CURRENT_POLICY = DEFAULT_POLICY
CURRENT_AGENT = "bob"
CURRENT_GOAL = None
CURRENT_BUDGET = None
CURRENT_ROOM = None
CURRENT_STATUS = None
LAST_GUIDANCE_GRAPH = None
DYNAMIC_TOOL_NAMES = set()
OP_NAME_TO_NODE = {}
MAZE_BASE = "http://localhost:5001/ontologies/maze.ttl#"
Maze = Namespace(MAZE_BASE)
_POLICY_CONTEXT = {
    "graph": None,
    "base_url": None,
    "agent_budget": None,
    "maze_rooms": None,
}

_flask_app = Flask("exploration_guidance_state")


def register_policy(name, fn):
    POLICIES[name] = fn


def _policy_all(graph, node):
    return True


def _policy_explorable(graph, node):
    return (node, RDF.type, ExplorationOntology.ExplorableOperation) in graph


def _policy_no_fee(graph, node):
    sparql = """
    PREFIX guidance: <http://localhost:5001/ontologies/guidance.ttl/>
    PREFIX maze: <http://localhost:5001/ontologies/maze.ttl#>
    ASK {
      VALUES ?op { <OP_URI> }
      ?op guidance:hasDangerCause ?cause .
      ?cause a maze:Fee ;
             maze:toAgent ?owner .
      FILTER (?owner != IRI("AGENT_IRI"))
    }
    """
    agent_iri = f"{MAZE_BASE}{CURRENT_AGENT}"
    query = sparql.replace("OP_URI", str(node)).replace("AGENT_IRI", agent_iri)
    try:
        result = graph.query(query)
        return not bool(result.askAnswer)
    except Exception:
        return True


def _policy_no_trap_final(graph, node):
    sparql = """
    PREFIX guidance: <http://localhost:5001/ontologies/guidance.ttl/>
    PREFIX maze: <http://localhost:5001/ontologies/maze.ttl#>
    ASK {
      VALUES ?op { <OP_URI> }
      ?op guidance:hasDangerCause ?cause .
      ?cause a maze:Final .
      FILTER NOT EXISTS {
        ?op guidance:hasOutcome ?outcome .
        ?outcome a maze:Exit .
      }
    }
    """
    query = sparql.replace("OP_URI", str(node))
    try:
        result = graph.query(query)
        return not bool(result.askAnswer)
    except Exception:
        return True


def _policy_explorable_or_safe_non_explorable(graph, node):
    if _policy_explorable(graph, node):
        return True
    if (node, RDF.type, ExplorationOntology.NonExplorableOperation) in graph:
        return _policy_no_fee(graph, node) and _policy_no_trap_final(graph, node)
    return False


def _policy_explorable_exits(graph, node):
    if _policy_explorable(graph, node):
        return True
    for outcome in graph.objects(node, ExplorationOntology.Guidance["hasOutcome"]):
        if (outcome, RDF.type, Maze.Exit) in graph:
            return True
    return False


def _policy_context(graph: Graph):
    if _POLICY_CONTEXT["graph"] is not graph:
        _POLICY_CONTEXT["graph"] = graph
        _POLICY_CONTEXT["base_url"] = None
        _POLICY_CONTEXT["agent_budget"] = None
        _POLICY_CONTEXT["maze_rooms"] = None
    return _POLICY_CONTEXT


def _request_base_for_node(graph: Graph, node):
    request_uri = graph.value(node, HTTPOntology.requestURI)
    if request_uri is None:
        return None
    try:
        parsed = urlparse(str(request_uri))
    except Exception:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _fetch_maze_rooms(base_url: str):
    if not base_url:
        return None
    try:
        response = requests.get(f"{base_url}/maze", timeout=2)
        if response.ok:
            payload = response.json()
            rooms = payload.get("rooms")
            return rooms if isinstance(rooms, dict) else None
    except Exception:
        return None
    return None


def _fetch_agent_budget(base_url: str, agent: str):
    if not base_url or not agent:
        return None
    try:
        response = requests.get(
            f"{base_url}/status",
            params={"agent_id": agent},
            timeout=2,
        )
        if response.ok:
            payload = response.json()
            budget = payload.get("budget")
            return int(budget) if budget is not None else None
    except Exception:
        return None
    return None


def _room_id_from_uri(room_uri):
    if not isinstance(room_uri, URIRef):
        return None
    uri_str = str(room_uri)
    if "#" in uri_str:
        return uri_str.rsplit("#", 1)[-1]
    if "/" in uri_str:
        return uri_str.rsplit("/", 1)[-1]
    return None


def _fee_from_graph(graph: Graph, node):
    for cause in graph.objects(node, ExplorationOntology.hasDangerCause):
        if (cause, RDF.type, Maze.Fee) not in graph:
            continue
        amount = graph.value(cause, Maze.amount)
        owner = graph.value(cause, Maze.toAgent)
        try:
            fee_value = int(amount) if amount is not None else None
        except Exception:
            fee_value = None
        return fee_value, owner
    return None, None


def _policy_explorable_owned_or_affordable(graph, node):
    if _policy_explorable(graph, node):
        return True

    ctx = _policy_context(graph)
    base_url = ctx["base_url"] or _request_base_for_node(graph, node)
    if base_url and ctx["base_url"] is None:
        ctx["base_url"] = base_url

    rooms = ctx["maze_rooms"]
    if rooms is None and base_url:
        rooms = _fetch_maze_rooms(base_url)
        ctx["maze_rooms"] = rooms

    budget = CURRENT_BUDGET
    if budget is None:
        budget = ctx["agent_budget"]
    if budget is None and base_url:
        budget = _fetch_agent_budget(base_url, CURRENT_AGENT)
        ctx["agent_budget"] = budget

    room_uri = graph.value(node, Maze.toRoom)
    room_id = _room_id_from_uri(room_uri)
    if room_id and rooms and room_id in rooms:
        room = rooms[room_id]
        if room.get("type") == "owned":
            owner = room.get("owner")
            fee = room.get("fee", 0)
            if owner == CURRENT_AGENT:
                return True
            if budget is None:
                return False
            return budget > int(fee)
        return False

    fee_amount, owner_iri = _fee_from_graph(graph, node)
    if fee_amount is None or budget is None:
        return False
    agent_iri = URIRef(f"{MAZE_BASE}{CURRENT_AGENT}")
    if owner_iri == agent_iri:
        return True
    return budget > fee_amount


register_policy("all", _policy_all)
register_policy("explorable", _policy_explorable)
register_policy("no_fee", _policy_no_fee)
register_policy("no_trap_final", _policy_no_trap_final)
register_policy(
    "explorable_or_safe_non_explorable", _policy_explorable_or_safe_non_explorable
)
register_policy(
    "explorable_or_owned_or_affordable", _policy_explorable_owned_or_affordable
)
register_policy("explorable_exits", _policy_explorable_exits)

if DEFAULT_POLICY not in POLICIES:
    CURRENT_POLICY = "all"


def _ontology_base_for_uri(uri: str):
    parsed = urlparse(uri)
    path = parsed.path or ""
    lower_path = path.lower()
    idx = lower_path.find(".ttl")
    if idx == -1:
        return None
    base_path = path[: idx + len(".ttl")]
    base_parsed = parsed._replace(path=base_path, params="", query="", fragment="")
    return base_parsed.geturl()


def _local_ontology_path(base_uri: str):
    parsed = urlparse(base_uri)
    relative = parsed.path.strip("/")
    if not relative:
        return None
    candidate = root / relative
    return candidate if candidate.exists() else None


def _parse_rdf_source(graph: Graph, source: str):
    try:
        graph.parse(source, format="turtle")
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _ontology_graph_for_base(base_uri: str):
    graph = Graph()
    if _parse_rdf_source(graph, base_uri):
        return graph
    local_path = _local_ontology_path(base_uri)
    if local_path:
        graph = Graph()
        if _parse_rdf_source(graph, local_path.as_uri()):
            return graph
    return None


def _get_resource_comment(resource, graph: Graph):
    comment = graph.value(resource, RDFS.comment)
    if isinstance(comment, Literal):
        text = str(comment).strip()
        if text:
            return text
    base = _ontology_base_for_uri(str(resource))
    if not base:
        return None
    ontology = _ontology_graph_for_base(base)
    if ontology is None:
        return None
    comment = ontology.value(resource, RDFS.comment)
    if isinstance(comment, Literal):
        text = str(comment).strip()
        if text:
            return text
    return None


@mcp.tool()
def http_request(url: str) -> str:
    """A tool to perform a GET HTTP request at the URL provided as parameter"""

    r = requests.get(url)
    guidance_info_url = get_guidance_link(r)
    print("guidance info url: ", guidance_info_url)
    if guidance_info_url is not None:
        print("has guidance url")
        g = Graph()
        g.parse(guidance_info_url)
        _apply_guidance(g)

    return r.text


def get_guidance_link(response):
    """
    Given a `requests` Response object, return the URL from the Link header
    whose rel="guidance". Return None if not present.
    """
    link_header = response.headers.get("Link")
    if not link_header:
        return None

    # Split on commas that separate link entries
    parts = [p.strip() for p in link_header.split(",")]

    # Regex to capture: <url>; rel="something"
    pattern = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')

    for part in parts:
        match = pattern.match(part)
        if match:
            url, rel = match.groups()
            if rel == "guidance":
                return url

    return None


def extract_operations(g: Graph):
    operations = []
    name_to_node = {}

    def current_tool_names():
        names = set()
        for attr in ("tools", "_tools"):
            tool_obj = getattr(mcp, attr, None)
            if isinstance(tool_obj, dict):
                names.update(tool_obj.keys())
            elif isinstance(tool_obj, (list, tuple, set)):
                for t in tool_obj:
                    if isinstance(t, str):
                        names.add(t)
                    elif isinstance(t, dict) and "name" in t:
                        names.add(t["name"])
                    elif hasattr(t, "name"):
                        names.add(t.name)
        return names

    def derive_name(rdf_node):
        label = g.value(rdf_node, RDFS.label)
        if isinstance(label, Literal) and str(label).strip():
            return str(label).strip()

        if isinstance(rdf_node, URIRef):
            uri_str = str(rdf_node).rstrip("/#")
            if "#" in uri_str:
                return uri_str.rsplit("#", 1)[-1]
            if "/" in uri_str:
                return uri_str.rsplit("/", 1)[-1]
            return uri_str
        return None

    def ensure_unique(name, existing):
        if name not in existing:
            return name
        idx = 2
        while f"{name}{idx}" in existing:
            idx += 1
        return f"{name}{idx}"

    operation_nodes = set()
    for op_type in (
        ExplorationOntology.Operation,
        ExplorationOntology.ExplorableOperation,
        ExplorationOntology.NonExplorableOperation,
        HTTPOntology.Request,
    ):
        for subj in g.subjects(RDF.type, op_type):
            operation_nodes.add(subj)

    existing_names = current_tool_names()

    policy_fn = POLICIES.get(CURRENT_POLICY, _policy_all)

    for node in operation_nodes:
        if not policy_fn(g, node):
            continue
        name_candidate = derive_name(node)
        if not name_candidate:
            name_candidate = generate_id()

        unique_name = ensure_unique(name_candidate, existing_names)
        existing_names.add(unique_name)

        op = Operation()
        op._name = unique_name
        name_to_node[unique_name] = node

        description_parts = []
        mode = TOOL_DESCRIPTION_MODE
        is_explorable = (node, RDF.type, ExplorationOntology.ExplorableOperation) in g
        node_types = set(g.objects(node, RDF.type))
        minimal_operation_semantics = (
            (node, RDF.type, ExplorationOntology.Operation) in g
            and (node, RDF.type, HTTPOntology.Request) in g
            and (node, RDF.type, ExplorationOntology.ExplorableOperation) not in g
            and (node, RDF.type, ExplorationOntology.NonExplorableOperation) not in g
            and not any(g.objects(node, ExplorationOntology.Guidance["hasOutcome"]))
            and not any(g.objects(node, ExplorationOntology.hasDangerCause))
            and node_types.issubset(
                {
                    ExplorationOntology.Operation,
                    HTTPOntology.Request,
                }
            )
        )
        outcome_comments = []
        for outcome in g.objects(node, ExplorationOntology.Guidance["hasOutcome"]):
            outcome_comment = _get_resource_comment(outcome, g)
            if outcome_comment:
                outcome_comments.append(outcome_comment)

        if mode in ("all", "no outcome"):
            desc_literal = g.value(node, RDFS.comment)
            if desc_literal:
                literal_text = str(desc_literal).strip()
                if literal_text:
                    description_parts.append(literal_text)

            class_uri = None
            if (node, RDF.type, ExplorationOntology.ExplorableOperation) in g:
                class_uri = ExplorationOntology.ExplorableOperation
            elif (node, RDF.type, ExplorationOntology.NonExplorableOperation) in g:
                class_uri = ExplorationOntology.NonExplorableOperation

            if class_uri:
                class_comment = _get_resource_comment(class_uri, g)
                if class_comment:
                    description_parts.append(class_comment)

            danger_comments = []
            print("node: ", node)
            for cause in g.objects(node, ExplorationOntology.hasDangerCause):
                print("cause: ", cause)
                cause_comment = _get_resource_comment(cause, g)
                if cause_comment:
                    danger_comments.append(cause_comment)
                else:
                    fallback = derive_name(cause)
                    danger_comments.append(fallback if fallback else str(cause))

            if danger_comments:
                description_parts.append("Danger causes: " + "; ".join(danger_comments))

        if mode in ("all", "outcome") and outcome_comments:
            label = "Outcome" if len(outcome_comments) == 1 else "Outcomes"
            description_parts.append(f"{label}: " + "; ".join(outcome_comments))

        if minimal_operation_semantics:
            op._description = "This is an operation"
        elif mode == "none":
            op._description = "This is an operation"
        elif mode == "explorability":
            op._description = (
                "Operation is explorable."
                if is_explorable
                else "Operation is not explorable."
            )
        elif description_parts:
            op._description = " ".join(description_parts)
        else:
            op._description = f"Operation {unique_name}"
        print("description for operation: ", op.description)

        method_literal = g.value(node, HTTPOntology.methodName)
        op.method = str(method_literal) if method_literal else "GET"

        request_uri = g.value(node, HTTPOntology.requestURI)
        if request_uri is not None:
            op.url = str(request_uri)

        body_literal = g.value(node, HTTPOntology.body)
        if body_literal is not None:
            op.body = str(body_literal)

        operations.append(op)

    return operations, name_to_node


def get_name(o):
    return o.name


def generate_id():
    return str(uuid.uuid4())


def get_description(o):
    return o.description


def get_function(o: Operation):
    def f():
        r = perform_operation(o)
        _sync_state_from_http_response(r)
        guidance_info_url = get_guidance_link(r)
        print("guidance info url: ", guidance_info_url)
        if guidance_info_url is not None:
            print("has guidance url")
            g = Graph()
            g.parse(guidance_info_url)
            _apply_guidance(g)

        return _response_payload(r)

    return f


def _response_payload(response: requests.Response):
    payload = {
        "status_code": response.status_code,
        "ok": response.ok,
    }

    content_type = response.headers.get("Content-Type", "").lower()
    if "application/json" in content_type:
        try:
            payload["content"] = response.json()
            return payload
        except ValueError:
            # Fallback to text when JSON parsing fails despite content-type.
            pass

    payload["content"] = response.text
    return payload


def perform_operation(o: Operation):
    method = getattr(o, "method", "GET") or "GET"
    url = getattr(o, "url", None)
    if url is None:
        raise ValueError("Operation missing request URL")

    body = getattr(o, "body", None)

    request_kwargs = {}
    if body is not None:
        try:
            json_body = json.loads(body)
        except Exception:
            request_kwargs["data"] = body
        else:
            request_kwargs["json"] = json_body

    response = requests.request(method.upper(), url, **request_kwargs)
    return response


def _sync_state_from_payload(payload):
    global CURRENT_AGENT
    global CURRENT_GOAL
    global CURRENT_BUDGET
    global CURRENT_ROOM
    global CURRENT_STATUS

    if not isinstance(payload, dict):
        return

    if "agent_id" in payload and payload["agent_id"] not in (None, ""):
        CURRENT_AGENT = str(payload["agent_id"])
        _POLICY_CONTEXT["agent_budget"] = CURRENT_BUDGET
    if "goal" in payload:
        goal = payload["goal"]
        CURRENT_GOAL = str(goal) if goal not in (None, "") else None
    if "budget" in payload:
        budget = payload["budget"]
        if budget in (None, ""):
            CURRENT_BUDGET = None
        else:
            try:
                CURRENT_BUDGET = int(budget)
            except Exception:
                pass
        _POLICY_CONTEXT["agent_budget"] = CURRENT_BUDGET
    if "room" in payload:
        room = payload["room"]
        CURRENT_ROOM = str(room) if room not in (None, "") else None
    if "status" in payload:
        status = payload["status"]
        CURRENT_STATUS = str(status) if status not in (None, "") else None


def _sync_state_from_http_response(response: requests.Response):
    try:
        payload = response.json()
    except Exception:
        return
    _sync_state_from_payload(payload)


def _clear_dynamic_tools():
    for name in list(DYNAMIC_TOOL_NAMES):
        try:
            mcp.remove_tool(name)
        except Exception:
            pass
    DYNAMIC_TOOL_NAMES.clear()


def _apply_guidance(g: Graph):
    global LAST_GUIDANCE_GRAPH
    global OP_NAME_TO_NODE
    LAST_GUIDANCE_GRAPH = g
    _clear_dynamic_tools()
    operations, name_to_node = extract_operations(g)
    OP_NAME_TO_NODE = name_to_node
    for o in operations:
        mcp.add_tool(get_function(o), name=get_name(o), description=get_description(o))
        DYNAMIC_TOOL_NAMES.add(get_name(o))


@mcp.tool()
def read_info(name: str) -> str:
    """Return the full exploration guidance Turtle for a named operation tool."""
    if not name:
        return "Operation name cannot be empty."
    if LAST_GUIDANCE_GRAPH is None:
        return "No exploration guidance graph is available yet."
    if name not in OP_NAME_TO_NODE:
        return f"No operation named '{name}' found."
    return LAST_GUIDANCE_GRAPH.serialize(format="turtle")


def set_policy(policy: str) -> str:
    """Set the tool visibility policy for operations (e.g., 'all', 'explorable')."""
    global CURRENT_POLICY
    if policy not in POLICIES:
        return f"Unknown policy '{policy}'. Available: {', '.join(sorted(POLICIES.keys()))}"
    CURRENT_POLICY = policy
    return f"Policy set to '{policy}'."


if SHOW_SET_POLICY_TOOL:
    mcp.add_tool(set_policy, name="set_policy", description=set_policy.__doc__)


@mcp.tool()
def set_agent(agent: str) -> str:
    """Set the active agent name used by policy evaluation (default: bob)."""
    global CURRENT_AGENT
    if not agent:
        return "Agent name cannot be empty."
    CURRENT_AGENT = agent
    _POLICY_CONTEXT["agent_budget"] = CURRENT_BUDGET
    return f"Agent set to '{agent}'."


@mcp.tool()
def set_goal(goal: str) -> str:
    """Set the active agent goal (e.g., 'exit_north')."""
    global CURRENT_GOAL
    if not goal:
        return "Goal cannot be empty."
    CURRENT_GOAL = str(goal)
    return f"Goal set to '{CURRENT_GOAL}'."


def set_budget(budget: str) -> str:
    """Legacy tool. Budget is managed via HTTP state GUI/API, not this tool."""
    return (
        "Budget updates are disabled in this tool. "
        "Use the HTTP GUI or PATCH /state with {'budget': <int|null>}."
    )


def _set_set_budget_tool_visibility(visible: bool):
    global SHOW_SET_BUDGET_TOOL
    SHOW_SET_BUDGET_TOOL = bool(visible)
    if SHOW_SET_BUDGET_TOOL:
        try:
            mcp.add_tool(set_budget, name="set_budget", description=set_budget.__doc__)
        except Exception:
            pass
    else:
        try:
            mcp.remove_tool("set_budget")
        except Exception:
            pass


def _serialize_state():
    return {
        "policy": CURRENT_POLICY,
        "agent": CURRENT_AGENT,
        "goal": CURRENT_GOAL,
        "budget": CURRENT_BUDGET,
        "room": CURRENT_ROOM,
        "status": CURRENT_STATUS,
        "tool_description_mode": TOOL_DESCRIPTION_MODE,
        "show_set_policy_tool": SHOW_SET_POLICY_TOOL,
        "show_set_budget_tool": SHOW_SET_BUDGET_TOOL,
        "available_policies": sorted(POLICIES.keys()),
        "has_guidance_graph": LAST_GUIDANCE_GRAPH is not None,
    }


@_flask_app.get("/state")
def get_state():
    return jsonify(_serialize_state())


@_flask_app.patch("/state")
def update_state():
    global CURRENT_POLICY
    global CURRENT_AGENT
    global CURRENT_GOAL
    global CURRENT_BUDGET
    global CURRENT_ROOM
    global CURRENT_STATUS
    global TOOL_DESCRIPTION_MODE

    payload = request.get_json(silent=True) or {}

    if "policy" in payload:
        policy = payload["policy"]
        if policy not in POLICIES:
            return jsonify(
                {"error": "Unknown policy.", "available": sorted(POLICIES.keys())}
            ), 400
        CURRENT_POLICY = policy

    if "agent" in payload:
        agent = payload["agent"]
        if agent:
            CURRENT_AGENT = str(agent)
            _POLICY_CONTEXT["agent_budget"] = CURRENT_BUDGET
        else:
            return jsonify({"error": "Agent name cannot be empty."}), 400

    if "goal" in payload:
        goal = payload["goal"]
        CURRENT_GOAL = str(goal) if goal not in (None, "") else None

    if "budget" in payload:
        budget = payload["budget"]
        if budget in (None, ""):
            CURRENT_BUDGET = None
        else:
            try:
                CURRENT_BUDGET = int(budget)
            except Exception:
                return jsonify({"error": "Budget must be an integer."}), 400
        _POLICY_CONTEXT["agent_budget"] = CURRENT_BUDGET

    if "room" in payload:
        room = payload["room"]
        CURRENT_ROOM = str(room) if room not in (None, "") else None

    if "status" in payload:
        status = payload["status"]
        CURRENT_STATUS = str(status) if status not in (None, "") else None

    if "tool_description_mode" in payload:
        mode = payload["tool_description_mode"]
        if mode not in ("all", "outcome", "none", "explorability", "no outcome"):
            return jsonify(
                {
                    "error": (
                        "tool_description_mode must be one of: "
                        "'all', 'outcome', 'none', 'explorability', 'no outcome'."
                    )
                }
            ), 400
        TOOL_DESCRIPTION_MODE = mode

    if "show_set_budget_tool" in payload:
        visible = payload["show_set_budget_tool"]
        if not isinstance(visible, bool):
            return jsonify({"error": "show_set_budget_tool must be boolean."}), 400
        _set_set_budget_tool_visibility(visible)

    return jsonify(_serialize_state())


def _run_flask_server():
    _flask_app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)


def _start_flask_thread():
    thread = threading.Thread(
        target=_run_flask_server, name="guidance-flask", daemon=True
    )
    thread.start()


if __name__ == "__main__":
    # Streamable HTTP on one endpoint
    _set_set_budget_tool_visibility(SHOW_SET_BUDGET_TOOL)
    _start_flask_thread()
    mcp.run(transport="streamable-http")
