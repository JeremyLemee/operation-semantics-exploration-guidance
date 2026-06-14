import json
import os
from pathlib import Path
import uuid

import HTTPOntology
from exploration_guidance_general import Model

from rdflib import Graph, URIRef, RDF, Literal, Namespace, BNode
from rdflib.namespace import RDFS

from ExplorationOntology import (
    Operation,
    ExplorableOperation,
    NonExplorableOperation,
    hasDangerCause,
)


def generate_uuid():
    return str(uuid.uuid4())


MAZE_FILE = Path(__file__).with_name(os.environ.get("MAZE_FILE", "maze.json"))
Maze = Namespace("http://localhost:5001/ontologies/maze.ttl#")
Guidance = Namespace("http://localhost:5001/ontologies/guidance.ttl/")
REGISTERED_AGENT_BUDGET = 10


class MazeModel(Model):
    _POLICY_ALIASES = {
        "all": "all",
        "full_guidance": "all",
        "full guidance": "all",
        "none": "none",
        "no_guidance": "none",
        "no guidance": "none",
        "explorability": "explorability",
        "outcome": "outcome",
        "outcome_only": "outcome_only",
        "outcome only": "outcome_only",
        "danger": "danger",
    }
    _VALID_POLICIES = set(_POLICY_ALIASES.values())

    def initialize(self, **kwargs):
        with MAZE_FILE.open(encoding="utf-8") as file:
            maze_definition = json.load(file)

        self._rooms = maze_definition["rooms"]
        self._doors = maze_definition["doors"]
        self._start_room = maze_definition.get("start_room", "start")
        self._default_budget = maze_definition.get("default_budget", REGISTERED_AGENT_BUDGET)
        self._default_goal = maze_definition.get("default_goal")

        self._agents = {}
        self._locked_doors = set()

        self._maze_uri = "http://localhost:5001/"
        self._valid_paths = {"/maze", "/status", "/register", "/reset", "/gui"}
        self._policy = "all"

    def set_policy(self, policy: str) -> None:
        normalized_policy = self._POLICY_ALIASES.get(policy.strip().lower())
        if normalized_policy is None:
            valid = ", ".join(sorted(self._POLICY_ALIASES))
            raise ValueError(f"Invalid policy '{policy}'. Expected one of: {valid}.")
        self._policy = normalized_policy

    def get_policy(self) -> str:
        return self._policy

    def _register_agent(self, agent_id):
        self._agents[agent_id] = {
            "room": self._start_room,
            "budget": self._default_budget,
            "subscription": True,
            "goal": self._default_goal,
            "status": "active",
        }
        return self._agents[agent_id]

    def _room_type(self, room_id):
        return self._rooms[room_id].get("type", "normal")

    def _update_agent_status(self, agent, room_id):
        room_type = self._room_type(room_id)
        if agent["budget"] < 0:
            agent["status"] = "bankrupt"
        elif room_type == "trap":
            agent["status"] = "trapped"
        elif room_type == "exit":
            agent["status"] = "exited"
        else:
            agent["status"] = "active"

    def _door_connects(self, door, room_id):
        return door["a"] == room_id or door["b"] == room_id

    def _other_side(self, door, room_id):
        if door["a"] == room_id:
            return door["b"]
        if door["b"] == room_id:
            return door["a"]
        return None

    def _potential_exits(self, door, source_room, target_room):
        potential_exits = door.get("potential_exits", [])
        if isinstance(potential_exits, list):
            return potential_exits
        if not isinstance(potential_exits, dict):
            return []
        if door["a"] == source_room and door["b"] == target_room:
            return potential_exits.get("a_to_b", [])
        if door["b"] == source_room and door["a"] == target_room:
            return potential_exits.get("b_to_a", [])
        return []

    def _extract_agent_id(self, req):
        agent_id = req.args.get("agent_id") or req.args.get("name")
        if agent_id:
            return agent_id
        data = req.get_json(silent=True)
        if isinstance(data, dict):
            payload_agent_id = data.get("agent_id") or data.get("name")
            if isinstance(payload_agent_id, str) and payload_agent_id:
                return payload_agent_id
        return "bob"

    def _apply_register(self, req, agent):
        data = req.get_json(silent=True)
        if not isinstance(data, dict):
            return
        if "budget" in data:
            agent["budget"] = int(data["budget"])
        if "subscription" in data:
            agent["subscription"] = bool(data["subscription"])
        if "goal" in data:
            goal = data["goal"]
            if goal in self._rooms:
                agent["goal"] = goal
        self._update_agent_status(agent, agent["room"])

    def _add_register_operation(self, graph, agent_id):
        op = URIRef(Maze[f"register_{agent_id}"])
        graph.add((op, RDF.type, HTTPOntology.Request))
        graph.add((op, RDF.type, Operation))
        graph.add((op, RDF.type, ExplorableOperation))
        graph.add((op, HTTPOntology.methodName, Literal("POST")))
        graph.add((op, HTTPOntology.requestURI, URIRef(self._maze_uri + "register")))
        graph.add(
            (
                op,
                HTTPOntology.body,
                Literal(json.dumps({"name": agent_id, "subscription": True})),
            )
        )
        graph.add((op, RDFS.comment, Literal(f"Register agent '{agent_id}' in the maze.")))

    def _room_name(self, room_id):
        room = self._rooms.get(room_id, {})
        return room.get("name", room_id)

    def _room_fee(self, room_id):
        return int(self._rooms[room_id].get("fee", 0))

    def _room_fee_requires_no_subscription(self, room_id):
        return bool(self._rooms[room_id].get("fee_requires_no_subscription", False))

    def _room_fee_applies(self, agent, room_id):
        fee = self._room_fee(room_id)
        if fee <= 0:
            return False
        requires_no_subscription = self._room_fee_requires_no_subscription(room_id)
        has_subscription = bool(agent.get("subscription", False))
        return not (requires_no_subscription and has_subscription)

    def _is_supported_path(self, path: str) -> bool:
        return path in self._valid_paths or path.startswith("/rooms/")

    def _apply_move(self, req, agent, agent_id):
        data = req.get_json(silent=True)
        if not isinstance(data, dict):
            return
        target_room = req.path.removeprefix("/rooms/")
        if not target_room or target_room not in self._rooms:
            return
        door_id = None
        door = None
        for candidate_door_id, candidate_door in self._doors.items():
            if {candidate_door["a"], candidate_door["b"]} == {agent["room"], target_room}:
                door_id = candidate_door_id
                door = candidate_door
                break
        if door_id is None or door is None or door_id in self._locked_doors:
            return
        if self._room_fee_applies(agent, target_room):
            agent["budget"] -= self._room_fee(target_room)
        agent["room"] = target_room
        if door["locks_after_use"]:
            self._locked_doors.add(door_id)
        self._update_agent_status(agent, agent["room"])

    def _add_fee_cause(self, graph, op_uri, room_id, agent):
        fee_node = BNode()
        fee = self._room_fee(room_id)
        requires_no_subscription = self._room_fee_requires_no_subscription(room_id)
        graph.add((fee_node, RDF.type, Maze.Fee))
        graph.add((fee_node, Maze.amount, Literal(fee)))
        if requires_no_subscription:
            graph.add((fee_node, Maze.requiresNoSubscription, Literal(True)))
            fee_comment = (
                f"Entering room '{self._room_name(room_id)}' ({room_id}) costs {fee} "
                "only if the agent does not have a subscription."
            )
        else:
            fee_comment = f"Entering room '{self._room_name(room_id)}' ({room_id}) costs {fee}."
        graph.add((fee_node, RDFS.comment, Literal(fee_comment)))
        graph.add((op_uri, hasDangerCause, fee_node))

    def _add_final_cause(self, graph, op_uri, room_id, room_type):
        final_node = BNode()
        graph.add((final_node, RDF.type, Maze.Final))
        final_comment = (
            f"Moving leads to a final {room_type} room '{self._room_name(room_id)}' ({room_id})."
        )
        graph.add((final_node, RDFS.comment, Literal(final_comment)))
        graph.add((op_uri, hasDangerCause, final_node))

    def _add_locked_cause(self, graph, op_uri, door_id, reason):
        locked_node = BNode()
        graph.add((locked_node, RDF.type, Maze.OneWayDoor))
        if reason == "locked_after_use":
            locked_comment = f"Door '{door_id}' will lock after use."
        else:
            locked_comment = f"Door '{door_id}' is currently locked."
        graph.add((locked_node, RDFS.comment, Literal(locked_comment)))
        graph.add((op_uri, hasDangerCause, locked_node))

    def _add_outcome(self, graph, op_uri, room_id, room_type, agent_id, potential_exits):
        outcome = BNode()
        outcome_type = Maze.Exit if room_type == "exit" else Maze.Move
        graph.add((outcome, RDF.type, outcome_type))
        graph.add((outcome, Maze.toRoom, URIRef(Maze[room_id])))
        outcome_comment = (
            f"Agent '{agent_id}' will be in room '{self._room_name(room_id)}' ({room_id})."
        )
        if isinstance(potential_exits, list) and potential_exits:
            exits_text = ", ".join(
                f"{self._room_name(str(exit_id))} ({exit_id})" for exit_id in potential_exits
            )
            outcome_comment += f" Potentially reachable exits: {exits_text}."
        graph.add((outcome, RDFS.comment, Literal(outcome_comment)))
        graph.add((op_uri, Guidance.hasOutcome, outcome))

    def process(self, req, response):
        uri_id = generate_uuid()
        g = Graph()

        if not self._is_supported_path(req.path):
            self._info[uri_id] = g
            return {"link": self._base_url + uri_id}

        agent_id = self._extract_agent_id(req)
        agent = self._agents.get(agent_id)

        if req.path == "/reset" and response.status_code < 400:
            self._agents.clear()
            self._locked_doors.clear()
            agent = None
        elif req.path == "/register" and response.status_code < 400:
            if agent is None:
                agent = self._register_agent(agent_id)
            self._apply_register(req, agent)
        elif req.path.startswith("/rooms/") and response.status_code < 400:
            if agent is not None and agent["status"] == "active":
                self._apply_move(req, agent, agent_id)

        if agent is None:
            self._add_register_operation(g, agent_id)
            self._info[uri_id] = g
            return {"link": self._base_url + uri_id}

        if agent["status"] != "active":
            self._info[uri_id] = g
            return {"link": self._base_url + uri_id}

        for door_id, door in self._doors.items():
            if not self._door_connects(door, agent["room"]):
                continue
            if door_id in self._locked_doors:
                continue

            target_room = self._other_side(door, agent["room"])
            if target_room is None:
                continue
            causes: list[str] = []

            if door["locks_after_use"]:
                causes.append("locked_after_use")
            target_room_type = self._room_type(target_room)
            if target_room_type in {"exit", "trap"}:
                causes.append("final")
            if self._room_fee(target_room) > 0:
                causes.append("fee")

            op = URIRef(Maze[f"move_{door_id}"])
            g.add((op, RDF.type, HTTPOntology.Request))
            if self._policy in {"none", "outcome_only"}:
                g.add((op, RDF.type, Operation))
            else:
                g.add((op, RDF.type, Maze.Move))
            g.add((op, HTTPOntology.methodName, Literal("POST")))
            g.add((op, HTTPOntology.requestURI, URIRef(self._maze_uri + "rooms/" + target_room)))
            g.add(
                (
                    op,
                    HTTPOntology.body,
                    Literal('{"agent_id":"' + agent_id + '"}'),
                )
            )

            outcome_room_id = target_room
            if self._policy in {"all", "outcome", "outcome_only"}:
                outcome_room_type = self._room_type(outcome_room_id)
                self._add_outcome(
                    g,
                    op,
                    outcome_room_id,
                    outcome_room_type,
                    agent_id,
                    self._potential_exits(door, agent["room"], target_room),
                )

            if self._policy in {"all", "explorability", "outcome", "danger"}:
                if causes:
                    g.add((op, RDF.type, NonExplorableOperation))
                    if self._policy in {"all", "danger"}:
                        if "locked_after_use" in causes:
                            self._add_locked_cause(g, op, door_id, "locked_after_use")
                        if "final" in causes:
                            self._add_final_cause(g, op, target_room, target_room_type)
                        if "fee" in causes:
                            self._add_fee_cause(g, op, target_room, agent)
                else:
                    g.add((op, RDF.type, ExplorableOperation))

        self._info[uri_id] = g
        return {"link": self._base_url + uri_id}
