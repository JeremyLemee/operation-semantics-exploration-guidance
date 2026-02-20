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

import uuid


def generate_uuid():
    return str(uuid.uuid4())


Maze = Namespace("http://localhost:5001/ontologies/maze.ttl#")
Guidance = Namespace("http://localhost:5001/ontologies/guidance.ttl/")


class MazeModel(Model):
    _VALID_POLICIES = {"all", "explorability", "outcome", "outcome_only", "danger", "none"}

    def initialize(self, **kwargs):
        self._rooms = {
    "start": {"type": "normal"},
    "hall": {"type": "normal"},
    "trap": {"type": "trap"},
    "music_room": {"type": "normal"},
    "billard_room": {"type": "owned", "owner": "bob", "fee": 6},
    "ballroom": {"type": "normal"},
    "museum": {"type": "normal"},
    "workshop": {"type": "owned", "owner": "alice", "fee": 6},
    "gallery": {"type": "owned", "owner": "alice", "fee": 6},
    "library": {"type": "owned", "owner": "bob", "fee": 6},
    "armory": {"type": "normal"},
    "courtyard": {"type": "owned", "owner": "bob", "fee": 6},
    "exit1": {"type": "exit"},
    "exit2": {"type": "exit"},
    "exit3": {"type": "exit"},
    "exit4": {"type": "exit"},
}

        self._doors = {
    "d1":  {"a": "start",       "b": "hall",          "locks_after_use": False, "potential_exits": []},
    "d2": {"a": "hall",        "b": "trap",          "locks_after_use": False, "potential_exits": []},
    "d3":  {"a": "hall",        "b": "music_room",    "locks_after_use": False, "potential_exits": []},
    "d4": {"a": "music_room",  "b": "library",       "locks_after_use": False, "potential_exits": []},
    "d5":  {"a": "music_room",   "b": "ballroom", "locks_after_use": True, "potential_exits": ["exit3"]},
    "d6": {"a": "ballroom","b": "exit3",         "locks_after_use": False, "potential_exits": []},
    "d7": {"a": "hall",        "b": "museum",        "locks_after_use": False, "potential_exits": []},
    "d8": {"a": "museum",      "b": "exit1",         "locks_after_use": False, "potential_exits": []},
    "d9": {"a": "museum",      "b": "billard_room", "locks_after_use": False, "potential_exits": []},
    "d11": {"a": "billard_room","b": "workshop",     "locks_after_use": False, "potential_exits": []},
    "d12": {"a": "workshop",    "b": "exit2",         "locks_after_use": False, "potential_exits": []},
    "d13":  {"a": "gallery",     "b": "workshop",      "locks_after_use": False, "potential_exits": []},
    "d14": {"a": "gallery",     "b": "courtyard",       "locks_after_use": False, "potential_exits": []},
    "d15":  {"a": "library",     "b": "armory",        "locks_after_use": False, "potential_exits": []},
    "d16":  {"a": "armory",      "b": "courtyard",     "locks_after_use": False, "potential_exits": []},
    "d20": {"a": "courtyard",      "b": "exit4",         "locks_after_use": False, "potential_exits": []},
}

        self._default_budget = 10
        self._default_goal = "exit1"

        self._agents = {}
        self._locked_doors = set()

        self._maze_uri = "http://localhost:5001/"
        self._valid_paths = {"/maze", "/status", "/register", "/move", "/reset", "/gui"}
        self._policy = "all"

    def set_policy(self, policy: str) -> None:
        if policy not in self._VALID_POLICIES:
            valid = ", ".join(sorted(self._VALID_POLICIES))
            raise ValueError(f"Invalid policy '{policy}'. Expected one of: {valid}.")
        self._policy = policy

    def get_policy(self) -> str:
        return self._policy

    def _ensure_agent(self, agent_id):
        if agent_id not in self._agents:
            self._agents[agent_id] = {
                "room": "start",
                "budget": self._default_budget,
                "goal": self._default_goal,
                "status": "active",
            }
        return self._agents[agent_id]

    def _update_agent_status(self, agent, room_id):
        room_type = self._rooms[room_id]["type"]
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

    def _extract_agent_id(self, req):
        agent_id = req.args.get("agent_id")
        if agent_id:
            return agent_id
        data = req.get_json(silent=True)
        if isinstance(data, dict):
            payload_agent_id = data.get("agent_id")
            if isinstance(payload_agent_id, str) and payload_agent_id:
                return payload_agent_id
        return "bob"

    def _apply_register(self, req, agent):
        data = req.get_json(silent=True)
        if not isinstance(data, dict):
            return
        if "budget" in data:
            agent["budget"] = int(data["budget"])
        if "goal" in data:
            goal = data["goal"]
            if goal in self._rooms and self._rooms[goal]["type"] == "exit":
                agent["goal"] = goal
        self._update_agent_status(agent, agent["room"])

    def _apply_move(self, req, agent, agent_id):
        data = req.get_json(silent=True)
        if not isinstance(data, dict):
            return
        door_id = data.get("door_id")
        if not door_id or door_id not in self._doors:
            return
        if door_id in self._locked_doors:
            return
        door = self._doors[door_id]
        if not self._door_connects(door, agent["room"]):
            return
        target_room = self._other_side(door, agent["room"])
        if target_room is None:
            return
        room = self._rooms[target_room]
        if room["type"] == "owned" and room["owner"] != agent_id:
            agent["budget"] -= room["fee"]
        agent["room"] = target_room
        if door["locks_after_use"]:
            self._locked_doors.add(door_id)
        self._update_agent_status(agent, agent["room"])

    def _add_fee_cause(self, graph, op_uri, room_id, room):
        fee_node = BNode()
        graph.add((fee_node, RDF.type, Maze.Fee))
        graph.add((fee_node, Maze.amount, Literal(room["fee"])))
        graph.add((fee_node, Maze.toAgent, URIRef(Maze[room["owner"]])))
        fee_comment = (
            f"Entering room '{room_id}' costs {room['fee']} and is paid to {room['owner']}."
        )
        graph.add((fee_node, RDFS.comment, Literal(fee_comment)))
        graph.add((op_uri, hasDangerCause, fee_node))

    def _add_final_cause(self, graph, op_uri, room_id, room_type):
        final_node = BNode()
        graph.add((final_node, RDF.type, Maze.Final))
        final_comment = f"Moving leads to a final {room_type} room '{room_id}'."
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
        outcome_comment = f"Agent '{agent_id}' will be in room '{room_id}'."
        if isinstance(potential_exits, list) and potential_exits:
            exits_text = ", ".join(str(exit_id) for exit_id in potential_exits)
            outcome_comment += f" Potentially reachable exits: {exits_text}."
        graph.add((outcome, RDFS.comment, Literal(outcome_comment)))
        graph.add((op_uri, Guidance.hasOutcome, outcome))

    def process(self, req, response):
        uri_id = generate_uuid()
        g = Graph()

        if req.path not in self._valid_paths:
            self._info[uri_id] = g
            return {"link": self._base_url + uri_id}

        agent_id = self._extract_agent_id(req)
        agent = self._ensure_agent(agent_id)

        if req.path == "/reset" and response.status_code < 400:
            self._agents.clear()
            self._locked_doors.clear()
            agent = self._ensure_agent(agent_id)
        elif req.path == "/register" and response.status_code < 400:
            self._apply_register(req, agent)
        elif req.path == "/move" and response.status_code < 400:
            if agent["status"] == "active":
                self._apply_move(req, agent, agent_id)

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
            room = self._rooms[target_room]
            causes: list[str] = []

            if door["locks_after_use"]:
                causes.append("locked_after_use")
            if room["type"] in {"exit", "trap"}:
                causes.append("final")
            if room["type"] == "owned" and room["owner"] != agent_id:
                causes.append("fee")

            op = URIRef(Maze[f"move_{door_id}"])
            g.add((op, RDF.type, HTTPOntology.Request))
            if self._policy in {"none", "outcome_only"}:
                g.add((op, RDF.type, Operation))
            else:
                g.add((op, RDF.type, Maze.Move))
            g.add((op, HTTPOntology.methodName, Literal("POST")))
            g.add((op, HTTPOntology.requestURI, URIRef(self._maze_uri + "move")))
            g.add(
                (
                    op,
                    HTTPOntology.body,
                    Literal(
                        '{"agent_id":"' + agent_id + '","door_id":"' + door_id + '"}'
                    ),
                )
            )

            outcome_room_id = target_room
            if self._policy in {"all", "outcome", "outcome_only"}:
                outcome_room_type = self._rooms[outcome_room_id]["type"]
                self._add_outcome(
                    g,
                    op,
                    outcome_room_id,
                    outcome_room_type,
                    agent_id,
                    door.get("potential_exits"),
                )

            if self._policy in {"all", "explorability", "outcome", "danger"}:
                if causes:
                    g.add((op, RDF.type, NonExplorableOperation))
                    if self._policy in {"all", "danger"}:
                        if "locked_after_use" in causes:
                            self._add_locked_cause(g, op, door_id, "locked_after_use")
                        if "final" in causes:
                            self._add_final_cause(g, op, target_room, room["type"])
                        if "fee" in causes:
                            self._add_fee_cause(g, op, target_room, room)
                else:
                    g.add((op, RDF.type, ExplorableOperation))

        self._info[uri_id] = g
        return {"link": self._base_url + uri_id}
