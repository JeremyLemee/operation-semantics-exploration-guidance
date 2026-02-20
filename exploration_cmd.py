#!/usr/bin/env python3
import json
import sys
import textwrap
from typing import List, Mapping, Optional, Tuple

import requests
from rdflib import Graph, RDF, Namespace, URIRef


GUIDANCE = Namespace("http://localhost:5001/ontologies/guidance.ttl/")
HTTP = Namespace("https://www.w3.org/2011/http#")


def parse_guidance_link(headers: Mapping[str, str]) -> Optional[str]:
    link_header = headers.get("Link") or headers.get("link")
    if not link_header:
        return None
    # Simple parser for: <url>;rel="guidance"
    for part in link_header.split(","):
        if 'rel="guidance"' in part or "rel=guidance" in part:
            start = part.find("<")
            end = part.find(">")
            if start != -1 and end != -1 and end > start:
                return part[start + 1 : end].strip()
    return None


def fetch_guidance(url: str) -> Optional[Graph]:
    resp = requests.get(url, headers={"Accept": "text/turtle"})
    if resp.status_code >= 400:
        return None
    g = Graph()
    g.parse(data=resp.text, format="turtle")
    return g


def do_request(
    method: str, url: str, body: Optional[str]
) -> Tuple[requests.Response, Optional[Graph]]:
    headers = {}
    data = None
    if body:
        headers["Content-Type"] = "application/json"
        data = body
    resp = requests.request(method=method, url=url, headers=headers, data=data)
    guidance_url = parse_guidance_link(resp.headers)
    guidance_graph = fetch_guidance(guidance_url) if guidance_url else None
    return resp, guidance_graph


def list_operations(g: Graph) -> List[URIRef]:
    explorable = set(g.subjects(RDF.type, GUIDANCE.ExplorableOperation))
    nonexplorable = set(g.subjects(RDF.type, GUIDANCE.NonExplorableOperation))
    return sorted(
        [subject for subject in (explorable | nonexplorable) if isinstance(subject, URIRef)],
        key=str,
    )


def op_summary(g: Graph, op: URIRef) -> str:
    types = {str(t).split("/")[-1] for t in g.objects(op, RDF.type)}
    kinds = ", ".join(sorted(types))
    return kinds


def op_http(g: Graph, op: URIRef) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    method = next(iter(g.objects(op, HTTP.methodName)), None)
    uri = next(iter(g.objects(op, HTTP.requestURI)), None)
    body = next(iter(g.objects(op, HTTP.body)), None)
    return (
        str(method) if method else None,
        str(uri) if uri else None,
        str(body) if body else None,
    )


def show_op_details(g: Graph, op: URIRef) -> str:
    lines = [f"Operation: {op}"]
    lines.append("Types:")
    for t in sorted({str(t) for t in g.objects(op, RDF.type)}):
        lines.append(f"  - {t}")
    method, uri, body = op_http(g, op)
    lines.append("HTTP:")
    if method:
        lines.append(f"  method: {method}")
    if uri:
        lines.append(f"  uri: {uri}")
    if body:
        lines.append(f"  body: {body}")
    causes = list(g.objects(op, GUIDANCE.hasDangerCause))
    if causes:
        lines.append("Danger causes:")
        for c in causes:
            lines.append(f"  - {c}")
            for t in g.objects(c, RDF.type):
                lines.append(f"      type: {t}")
    outcomes = list(g.objects(op, GUIDANCE.hasOutcome))
    if outcomes:
        lines.append("Outcomes:")
        for o in outcomes:
            lines.append(f"  - {o}")
            for t in g.objects(o, RDF.type):
                lines.append(f"      type: {t}")
    return "\n".join(lines)


def print_status(resp: requests.Response):
    if resp.headers.get("Content-Type", "").startswith("application/json"):
        try:
            obj = resp.json()
            print(json.dumps(obj, indent=2))
            return
        except json.JSONDecodeError:
            pass
    print(resp.text)


def usage():
    msg = """
    Commands:
      help                         Show this help
      status                       Fetch /status and guidance
      ops                          List available operations
      show <idx>                   Show RDF-derived details for an operation
      turtle <idx>                 Show Turtle serialization for an operation
      run <idx>                    Execute the operation
      refresh                      Re-fetch /status and guidance
      quit                         Exit
    """
    print(textwrap.dedent(msg).strip())


def main():
    base_url = "http://localhost:5001"
    agent_id = "bob"
    if len(sys.argv) > 1:
        agent_id = sys.argv[1]

    print(f"Exploration CLI (proxy: {base_url}, agent: {agent_id})")
    print("Registering agent...")
    _, g = do_request(
        "POST",
        f"{base_url}/register",
        json.dumps(
            {
                "agent_id": agent_id,
                "budget": 10,
                "goal": "exit_north",
            }
        ),
    )
    guidance = g

    def refresh():
        nonlocal guidance
        resp, g = do_request("GET", f"{base_url}/status?agent_id={agent_id}", None)
        print_status(resp)
        if g:
            guidance = g
        if guidance:
            print(f"Guidance loaded with {len(list_operations(guidance))} operations.")
        else:
            print("No guidance available.")

    refresh()
    usage()

    while True:
        try:
            line = input("exploration> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]
        args = parts[1:]
        if cmd in {"quit", "exit"}:
            break
        if cmd == "help":
            usage()
            continue
        if cmd in {"status", "refresh"}:
            refresh()
            continue
        if cmd == "ops":
            if not guidance:
                print("No guidance loaded. Run 'refresh' first.")
                continue
            ops = list_operations(guidance)
            if not ops:
                print("No operations found.")
                continue
            for idx, op in enumerate(ops, 1):
                print(f"{idx}. {op} [{op_summary(guidance, op)}]")
            continue
        if cmd in {"show", "run", "turtle"}:
            if not guidance:
                print("No guidance loaded. Run 'refresh' first.")
                continue
            if not args:
                print("Missing operation index.")
                continue
            try:
                idx = int(args[0]) - 1
            except ValueError:
                print("Index must be a number.")
                continue
            ops = list_operations(guidance)
            if idx < 0 or idx >= len(ops):
                print("Invalid operation index.")
                continue
            op = ops[idx]
            if cmd == "show":
                print(show_op_details(guidance, op))
            elif cmd == "turtle":
                sub = Graph()
                for s, p, o in guidance.triples((op, None, None)):
                    sub.add((s, p, o))
                    if isinstance(o, URIRef):
                        continue
                    for s2, p2, o2 in guidance.triples((o, None, None)):
                        sub.add((s2, p2, o2))
                print(sub.serialize(format="turtle"))
            else:
                method, uri, body = op_http(guidance, op)
                if not method or not uri:
                    print("Operation missing HTTP method or URI.")
                    continue
                resp, g = do_request(method, uri, body)
                print_status(resp)
                if g:
                    guidance = g
            continue
        print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    main()
