import json
import os
from pathlib import Path

from flask import Flask, request, jsonify, abort

app = Flask(__name__)

# Path explorable: start -> hall -> museum -> exit1

MAZE_FILE = Path(__file__).with_name(os.environ.get("MAZE_FILE", "maze.json"))


def load_maze_definition():
    with MAZE_FILE.open(encoding="utf-8") as file:
        return json.load(file)


MAZE_DEFINITION = load_maze_definition()
ROOMS = MAZE_DEFINITION["rooms"]
DOORS = MAZE_DEFINITION["doors"]
START_ROOM = MAZE_DEFINITION.get("start_room", "start")
REGISTERED_AGENT_BUDGET = 10
DEFAULT_BUDGET = MAZE_DEFINITION.get("default_budget", REGISTERED_AGENT_BUDGET)
DEFAULT_AGENT_IDS = tuple(MAZE_DEFINITION.get("default_agent_ids", ()))


def build_default_agents():
    return {
        agent_id: {
            "room": START_ROOM,
            "budget": DEFAULT_BUDGET,
            "subscription": True,
            "status": "active",
        }
        for agent_id in DEFAULT_AGENT_IDS
    }


AGENTS = build_default_agents()
LOCKED_DOORS = set()


def room_data(room_id):
    return ROOMS[room_id]


def room_type(room_id):
    return room_data(room_id).get("type", "normal")


def room_name(room_id):
    return room_data(room_id).get("name", room_id)


def room_fee(room_id):
    return int(room_data(room_id).get("fee", 0))


def room_fee_requires_no_subscription(room_id):
    return bool(room_data(room_id).get("fee_requires_no_subscription", False))


def room_fee_applies(agent, room_id):
    fee = room_fee(room_id)
    if fee <= 0:
        return False
    if room_fee_requires_no_subscription(room_id) and agent.get("subscription", False):
        return False
    return True


def room_fee_details(room_id, agent=None):
    fee = room_fee(room_id)
    if fee <= 0:
        return None
    details = {
        "amount": fee,
        "requires_no_subscription": room_fee_requires_no_subscription(room_id),
    }
    if agent is not None:
        details["applies_to_agent"] = room_fee_applies(agent, room_id)
    return details


def ensure_agent(agent_id):
    if agent_id not in AGENTS:
        AGENTS[agent_id] = {
            "room": START_ROOM,
            "budget": DEFAULT_BUDGET,
            "subscription": True,
            "status": "active",
        }
    return AGENTS[agent_id]


def reset_agent(agent_id):
    AGENTS[agent_id] = {
        "room": START_ROOM,
        "budget": DEFAULT_BUDGET,
        "subscription": True,
        "status": "active",
    }
    return AGENTS[agent_id]


def agent_can_continue(agent):
    return agent["status"] == "active"


def agent_has_ended(agent):
    return not agent_can_continue(agent)


def door_connects(door, room_id):
    return door["a"] == room_id or door["b"] == room_id


def other_side(door, room_id):
    if door["a"] == room_id:
        return door["b"]
    if door["b"] == room_id:
        return door["a"]
    return None


def potential_exits_for_direction(door, source_room, target_room):
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


def find_door_between(room_a, room_b):
    for door_id, door in DOORS.items():
        if {door["a"], door["b"]} == {room_a, room_b}:
            return door_id, door
    return None, None


def has_available_affordance(room_id):
    return any(
        door_id not in LOCKED_DOORS and door_connects(door, room_id)
        for door_id, door in DOORS.items()
    )


def build_operation_semantics(door_id, door, source_room_id, target_room_id, agent_id):
    target_type = room_type(target_room_id)
    causes = []

    if door["locks_after_use"]:
        causes.append(
            {
                "type": "maze:LockedDoor",
                "detail": "Door locks after use",
                "door_id": door_id,
            }
        )

    if target_type in {"exit", "trap"}:
        causes.append(
            {
                "type": "maze:Final",
                "detail": f"Entering {target_type} room",
                "room_id": target_room_id,
            }
        )

    if room_fee_applies(AGENTS[agent_id], target_room_id):
        causes.append(
            {
                "type": "maze:Fee",
                "detail": "Entering room incurs a fee",
                "room_id": target_room_id,
                "fee": room_fee(target_room_id),
                "requires_no_subscription": room_fee_requires_no_subscription(target_room_id),
            }
        )

    operation = {
        "explorable": len(causes) == 0,
        "danger_causes": causes,
    }
    fee_details = room_fee_details(target_room_id, AGENTS.get(agent_id))
    if fee_details is not None:
        operation["fee"] = fee_details
    potential_exits = potential_exits_for_direction(door, source_room_id, target_room_id)
    if potential_exits:
        operation["potential_exits"] = list(potential_exits)
    return operation


def update_agent_status(agent, room_id):
    current_room_type = room_type(room_id)
    if agent["budget"] < 0:
        agent["status"] = "bankrupt"
    elif current_room_type == "trap":
        agent["status"] = "trapped"
    elif current_room_type == "exit":
        agent["status"] = "exited"
    elif not has_available_affordance(room_id):
        agent["status"] = "no_affordance"
    else:
        agent["status"] = "active"


def build_visual_representation():
    rooms = {
        room_id: {
            key: value
            for key, value in room_data.items()
            if key in {"name", "type", "fee", "fee_requires_no_subscription"} and value is not None
        }
        for room_id, room_data in ROOMS.items()
    }
    doors = []
    adjacency = {}

    for door_id, door in DOORS.items():
        room_a = door["a"]
        room_b = door["b"]

        if room_a not in rooms:
            rooms[room_a] = {}
        if room_b not in rooms:
            rooms[room_b] = {}

        adjacency.setdefault(room_a, []).append({"door_id": door_id, "to": room_b})
        adjacency.setdefault(room_b, []).append({"door_id": door_id, "to": room_a})
        doors.append(
            {
                "door_id": door_id,
                "from": room_a,
                "to": room_b,
                "locks_after_use": door["locks_after_use"],
                "locked": door_id in LOCKED_DOORS,
                "potential_exits": door.get("potential_exits", []),
            }
        )

    for room_id in rooms:
        adjacency.setdefault(room_id, [])

    return {"rooms": rooms, "doors": doors, "adjacency": adjacency}


@app.route("/maze", methods=["GET"])
def get_maze():
    return jsonify(
        {
            "rooms": ROOMS,
            "doors": DOORS,
            "locked_doors": sorted(LOCKED_DOORS),
        }
    )


@app.route("/visual", methods=["GET"])
def get_visual():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Maze Visual</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f4ef;
      --panel: #fffaf0;
      --ink: #2a2a2a;
      --border: #d2c8b6;
      --primary: #1c6b57;
      --secondary: #24558f;
      --danger: #ab3d3d;
      --muted: #70695e;
    }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top right, #fffef8 0%, var(--bg) 60%);
      color: var(--ink);
    }
    header {
      background: var(--panel);
      border-bottom: 2px solid var(--border);
      padding: 14px 18px;
    }
    h1, h2 {
      margin: 0 0 8px 0;
    }
    main {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 14px;
      padding: 14px;
    }
    section {
      background: var(--panel);
      border: 2px solid var(--border);
      border-radius: 10px;
      padding: 12px;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    input, button {
      padding: 7px 9px;
      border-radius: 7px;
      border: 1px solid var(--border);
      font-size: 14px;
    }
    button {
      border: 0;
      color: #fff;
      background: var(--primary);
      cursor: pointer;
    }
    button.secondary { background: var(--secondary); }
    button.danger { background: var(--danger); }
    button[disabled] { background: #aba9a3; cursor: not-allowed; }
    .agents {
      display: grid;
      gap: 8px;
      max-height: 420px;
      overflow: auto;
      padding-right: 4px;
    }
    .agent-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }
    .agent-card.selected {
      border-color: var(--secondary);
      box-shadow: 0 0 0 2px rgba(36, 85, 143, 0.16);
    }
    .muted { color: var(--muted); }
    .doors {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .pill {
      display: inline-block;
      border-radius: 999px;
      background: #ece5d6;
      padding: 3px 10px;
      font-size: 13px;
      margin-right: 6px;
      margin-bottom: 6px;
    }
    pre {
      margin: 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      max-height: 240px;
      overflow: auto;
      font-size: 12px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Maze Visual</h1>
    <div class="row">
      <input id="agentIdInput" placeholder="agent name (e.g. bob)" />
      <button id="registerBtn">Register Agent</button>
      <button id="resetBtn" class="danger">Reset Maze</button>
      <span id="flash" class="muted"></span>
    </div>
  </header>
  <main>
    <section>
      <h2>Registered Agents</h2>
      <div class="agents" id="agentsList"></div>
    </section>
    <section>
      <h2>Selected Agent</h2>
      <div id="selectedSummary" class="muted">Register and select an agent.</div>
      <h2>Affordances</h2>
      <div id="affordances" class="doors"></div>
      <h2>Last Action</h2>
      <pre id="lastAction">No actions yet.</pre>
    </section>
  </main>
<script>
  const state = {
    maze: null,
    registeredAgentIds: [],
    agentStates: {},
    selectedAgentId: null,
    lastAction: null,
  };

  async function apiJson(url, options) {
    const res = await fetch(url, options);
    let data = {};
    try {
      data = await res.json();
    } catch {
      data = { message: "Invalid JSON response" };
    }
    if (!res.ok) {
      throw new Error(data.message || `Request failed (${res.status})`);
    }
    return data;
  }

  function flash(message, isError = false) {
    const node = document.getElementById("flash");
    node.textContent = message;
    node.style.color = isError ? "#ab3d3d" : "#1c6b57";
    setTimeout(() => {
      if (node.textContent === message) {
        node.textContent = "";
      }
    }, 2500);
  }

  async function refreshMaze() {
    state.maze = await apiJson("/maze");
  }

  async function refreshAgents() {
    const updates = await Promise.all(
      state.registeredAgentIds.map(async (agentId) => {
        const status = await apiJson(`/status?agent_id=${encodeURIComponent(agentId)}`);
        return [agentId, status];
      })
    );
    updates.forEach(([agentId, status]) => {
      state.agentStates[agentId] = status;
    });
  }

  function uniquePush(list, value) {
    if (!list.includes(value)) {
      list.push(value);
    }
  }

  function selectedAgent() {
    return state.selectedAgentId ? state.agentStates[state.selectedAgentId] : null;
  }

  function renderAgents() {
    const list = document.getElementById("agentsList");
    list.innerHTML = "";
    if (!state.registeredAgentIds.length) {
      list.innerHTML = `<div class="muted">No registered agents.</div>`;
      return;
    }
    state.registeredAgentIds.forEach((agentId) => {
      const agent = state.agentStates[agentId];
      if (!agent) {
        return;
      }
      const card = document.createElement("div");
      card.className = `agent-card ${state.selectedAgentId === agentId ? "selected" : ""}`;
      card.innerHTML = `
        <div><strong>${agentId}</strong></div>
        <div class="pill">Room: ${agent.room_name || agent.room}</div>
        <div class="pill">Budget: ${agent.budget}</div>
        <div class="pill">Subscription: ${agent.subscription ? "yes" : "no"}</div>
        <div class="pill">Status: ${agent.status}</div>
      `;
      const selectBtn = document.createElement("button");
      selectBtn.className = "secondary";
      selectBtn.textContent = "Select";
      selectBtn.onclick = () => {
        state.selectedAgentId = agentId;
        render();
      };
      card.appendChild(selectBtn);
      list.appendChild(card);
    });
  }

  function renderSelectedSummary() {
    const node = document.getElementById("selectedSummary");
    const agent = selectedAgent();
    if (!agent) {
      node.textContent = "Register and select an agent.";
      return;
    }
    node.innerHTML = `
      <div class="pill">Agent: ${agent.agent_id}</div>
      <div class="pill">Room: ${agent.room_name || agent.room}</div>
      <div class="pill">Budget: ${agent.budget}</div>
      <div class="pill">Subscription: ${agent.subscription ? "yes" : "no"}</div>
      <div class="pill">Status: ${agent.status}</div>
    `;
  }

  function renderAffordances() {
    const box = document.getElementById("affordances");
    box.innerHTML = "";
    const agent = selectedAgent();
    if (!state.maze || !agent) {
      box.innerHTML = `<span class="muted">Select an agent to show moves.</span>`;
      return;
    }
    const locked = new Set(state.maze.locked_doors || []);
    const doors = Object.entries(state.maze.doors || {})
      .filter(([_, door]) => door.a === agent.room || door.b === agent.room);
    if (!doors.length) {
      box.innerHTML = `<span class="muted">No affordances from this room.</span>`;
      return;
    }
    doors.forEach(([doorId, door]) => {
      const target = door.a === agent.room ? door.b : door.a;
      const targetName = (state.maze.rooms?.[target]?.name || target);
      const isLocked = locked.has(doorId);
      const btn = document.createElement("button");
      btn.className = "secondary";
      const rawExits = door.potential_exits || [];
      const directedExits = Array.isArray(rawExits)
        ? rawExits
        : door.a === agent.room && door.b === target
          ? rawExits.a_to_b || []
          : door.b === agent.room && door.a === target
            ? rawExits.b_to_a || []
            : [];
      const exits = directedExits.length
        ? ` [exits: ${directedExits.join(", ")}]`
        : "";
      btn.textContent =
        `${doorId}: ${agent.room_name || agent.room} -> ${targetName}` +
        `${isLocked ? " (locked)" : ""}${exits}`;
      btn.disabled = isLocked || agent.status !== "active";
      btn.onclick = () => doMove(target);
      box.appendChild(btn);
    });
  }

  function renderLastAction() {
    document.getElementById("lastAction").textContent = state.lastAction
      ? JSON.stringify(state.lastAction, null, 2)
      : "No actions yet.";
  }

  function render() {
    renderAgents();
    renderSelectedSummary();
    renderAffordances();
    renderLastAction();
  }

  async function doMove(roomId) {
    const agentId = state.selectedAgentId;
    if (!agentId) {
      flash("Select an agent first.", true);
      return;
    }
    try {
      const result = await apiJson(`/rooms/${encodeURIComponent(roomId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: agentId }),
      });
      state.lastAction = result;
      await refreshMaze();
      await refreshAgents();
      flash(`Moved to ${result.room}`);
      render();
    } catch (err) {
      state.lastAction = { error: String(err) };
      renderLastAction();
      flash(String(err), true);
    }
  }

  async function registerAgent() {
    const agentId = document.getElementById("agentIdInput").value.trim();
    if (!agentId) {
      flash("Agent name is required.", true);
      return;
    }
    try {
      const result = await apiJson("/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: agentId }),
      });
      uniquePush(state.registeredAgentIds, agentId);
      state.agentStates[agentId] = result;
      state.selectedAgentId = agentId;
      state.lastAction = {
        message: `Registered ${agentId} at room ${result.room}`,
        room: result.room,
        room_name: result.room_name,
        budget: result.budget,
        status: result.status,
      };
      await refreshMaze();
      await refreshAgents();
      flash(`Registered ${agentId}`);
      render();
    } catch (err) {
      flash(String(err), true);
    }
  }

  async function resetMaze() {
    try {
      await apiJson("/reset", { method: "POST" });
      state.registeredAgentIds = [];
      state.agentStates = {};
      state.selectedAgentId = null;
      state.lastAction = { message: "Maze reset" };
      await refreshMaze();
      render();
      flash("Maze reset");
    } catch (err) {
      flash(String(err), true);
    }
  }

  document.getElementById("registerBtn").onclick = registerAgent;
  document.getElementById("resetBtn").onclick = resetMaze;

  (async () => {
    await refreshMaze();
    render();
  })();
</script>
</body>
</html>
"""


@app.route("/status", methods=["GET"])
def get_status():
    agent_id = request.args.get("agent_id") or request.args.get("name") or "bob"
    agent = AGENTS.get(agent_id)
    if agent is None:
        return jsonify({"error": f"Unknown agent '{agent_id}'."}), 404
    update_agent_status(agent, agent["room"])
    return jsonify(
        {
            "agent_id": agent_id,
            "room": agent["room"],
            "budget": agent["budget"],
            "subscription": bool(agent.get("subscription", False)),
            "status": agent["status"],
            "end": agent_has_ended(agent),
            "room_name": room_name(agent["room"]),
            "room_fee": room_fee_details(agent["room"], agent),
        }
    )


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "/register expects a JSON object body."}), 400

    agent_id = data.get("name") or data.get("agent_id")
    if not agent_id:
        abort(400, "Missing name")
    unexpected_fields = sorted(set(data.keys()) - {"name", "agent_id", "subscription"})
    if unexpected_fields:
        return (
            jsonify(
                {
                    "error": (
                        "Only 'name', 'agent_id', and 'subscription' "
                        "are allowed in /register payload."
                    ),
                    "unexpected_fields": unexpected_fields,
                }
            ),
            400,
        )
    if agent_id in AGENTS:
        return jsonify({"error": f"Agent '{agent_id}' is already registered."}), 400

    agent = {
        "room": START_ROOM,
        "budget": DEFAULT_BUDGET,
        "subscription": bool(data.get("subscription", True)),
        "status": "active",
    }
    AGENTS[agent_id] = agent
    return jsonify(
        {
            "agent_id": agent_id,
            "name": agent_id,
            "room": agent["room"],
            "budget": agent["budget"],
            "subscription": agent["subscription"],
            "status": agent["status"],
            "end": agent_has_ended(agent),
            "room_name": room_name(agent["room"]),
            "room_fee": room_fee_details(agent["room"], agent),
        }
    )


@app.route("/rooms/<room_id>", methods=["POST"])
def move_to_room(room_id):
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": f"/rooms/{room_id} expects a JSON object body."}), 400

    agent_id = data.get("agent_id") or data.get("name")
    if not agent_id:
        abort(400, "Missing agent_id")

    agent = AGENTS.get(agent_id)
    if agent is None:
        return jsonify({"message": f"Unknown agent '{agent_id}'."}), 404
    update_agent_status(agent, agent["room"])
    if agent["status"] != "active":
        return jsonify(
            {
                "message": f"Agent is {agent['status']} and cannot move",
                "agent_id": agent_id,
                "status": agent["status"],
                "end": agent_has_ended(agent),
            }
        ), 400

    if room_id not in ROOMS:
        abort(404, "Unknown room_id")

    door_id, door = find_door_between(agent["room"], room_id)
    if door is None:
        abort(400, "Target room is not reachable from agent's current room")

    if door_id in LOCKED_DOORS:
        return jsonify(
            {
                "message": "Door is locked",
                "agent_id": agent_id,
                "room": agent["room"],
                "status": agent["status"],
                "end": agent_has_ended(agent),
                "operation": {
                    "explorable": False,
                    "danger_causes": [
                        {
                            "type": "maze:LockedDoor",
                            "detail": "Door is locked",
                            "door_id": door_id,
                        }
                    ],
                },
            }
        ), 400

    operation = build_operation_semantics(door_id, door, agent["room"], room_id, agent_id)

    # Apply room entry effects
    if room_fee_applies(agent, room_id):
        agent["budget"] -= room_fee(room_id)

    agent["room"] = room_id

    if door["locks_after_use"]:
        LOCKED_DOORS.add(door_id)

    update_agent_status(agent, agent["room"])

    return jsonify(
        {
            "message": f"Moved to {room_name(room_id)} ({room_id})",
            "agent_id": agent_id,
            "room": agent["room"],
            "budget": agent["budget"],
            "subscription": bool(agent.get("subscription", False)),
            "status": agent["status"],
            "end": agent_has_ended(agent),
            "room_name": room_name(agent["room"]),
            "room_fee": room_fee_details(agent["room"], agent),
            "operation": operation,
        }
    )


@app.route("/<agent_id>/restart", methods=["POST"])
def restart(agent_id):
    agent = reset_agent(agent_id)
    return jsonify(
        {
            "agent_id": agent_id,
            "room": agent["room"],
            "budget": agent["budget"],
            "subscription": bool(agent.get("subscription", False)),
            "status": agent["status"],
            "end": agent_has_ended(agent),
            "room_name": room_name(agent["room"]),
            "room_fee": room_fee_details(agent["room"], agent),
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    AGENTS.clear()
    AGENTS.update(build_default_agents())
    LOCKED_DOORS.clear()
    return jsonify({"message": "Maze state reset"})


@app.route("/gui", methods=["GET"])
def gui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Maze Debug GUI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f3ec;
      --panel: #fff9f0;
      --border: #d9cfc0;
      --ink: #2b2b2b;
      --accent: #1f6b52;
      --warn: #b33a3a;
    }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background: linear-gradient(140deg, #f7f3ec 0%, #efe7da 100%);
      color: var(--ink);
    }
    header {
      padding: 16px 20px;
      border-bottom: 2px solid var(--border);
      background: var(--panel);
    }
    main {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      padding: 16px;
    }
    section {
      background: var(--panel);
      border: 2px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
    }
    h1, h2 {
      margin: 0 0 8px 0;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    label {
      font-weight: 600;
    }
    input, button, select {
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: white;
      border: 0;
    }
    button.secondary {
      background: #6b6b6b;
    }
    button.danger {
      background: var(--warn);
    }
    pre {
      background: #fff;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid var(--border);
      overflow: auto;
      max-height: 280px;
      font-size: 12px;
    }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e7dccb;
      font-size: 12px;
      margin-right: 6px;
    }
    .status {
      font-weight: 700;
    }
    .doors {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .door-btn {
      background: #204d87;
    }
    .door-btn[disabled] {
      background: #aaa;
      cursor: not-allowed;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Maze Debug GUI</h1>
    <div class="row">
      <label for="agentId">Agent Name</label>
      <input id="agentId" value="tester" />
      <button id="registerBtn">Register</button>
      <button id="resetBtn" class="danger">Reset Maze</button>
    </div>
    <div class="row">
      <span class="pill" id="roomPill">Room: -</span>
      <span class="pill" id="budgetPill">Budget: -</span>
      <span class="pill status" id="statusPill">Status: -</span>
    </div>
  </header>
  <main>
    <section>
      <h2>Available Moves</h2>
      <div id="doorButtons" class="doors"></div>
      <h2>Last Operation</h2>
      <pre id="lastOp">No moves yet.</pre>
    </section>
    <section>
      <h2>Maze State</h2>
      <pre id="mazeState">Loading...</pre>
      <h2>Agent State</h2>
      <pre id="agentState">Loading...</pre>
    </section>
  </main>
<script>
  const state = {
    maze: null,
    agent: null,
    lastOp: null,
  };

  async function fetchMaze() {
    const res = await fetch("/maze");
    state.maze = await res.json();
    renderMaze();
  }

  async function fetchAgent() {
    const agentId = getAgentId();
    const res = await fetch(`/status?agent_id=${encodeURIComponent(agentId)}`);
    state.agent = await res.json();
    renderAgent();
    renderMoves();
  }

  function renderMaze() {
    document.getElementById("mazeState").textContent = JSON.stringify(state.maze, null, 2);
  }

  function renderAgent() {
    document.getElementById("agentState").textContent = JSON.stringify(state.agent, null, 2);
    document.getElementById("roomPill").textContent =
      `Room: ${state.agent.room_name || state.agent.room}`;
    document.getElementById("budgetPill").textContent = `Budget: ${state.agent.budget}`;
    document.getElementById("statusPill").textContent = `Status: ${state.agent.status}`;
  }

  function renderMoves() {
    const container = document.getElementById("doorButtons");
    container.innerHTML = "";
    if (!state.agent || !state.maze) return;
    const room = state.agent.room;
    const locked = new Set(state.maze.locked_doors);
    const doors = Object.entries(state.maze.doors)
      .filter(([_, door]) => door.a === room || door.b === room);
    if (!doors.length) {
      container.textContent = "No exits from this room.";
      return;
    }
    doors.forEach(([doorId, door]) => {
      const btn = document.createElement("button");
      btn.className = "door-btn";
      const target = door.a === room ? door.b : door.a;
      const targetName = (state.maze.rooms?.[target]?.name || target);
      const lockedLabel = locked.has(doorId) ? " (locked)" : "";
      const rawExits = door.potential_exits || [];
      const directedExits = Array.isArray(rawExits)
        ? rawExits
        : door.a === room && door.b === target
          ? rawExits.a_to_b || []
          : door.b === room && door.a === target
            ? rawExits.b_to_a || []
            : [];
      const exits = directedExits.length
        ? ` [exits: ${directedExits.join(", ")}]`
        : "";
      btn.textContent = `${doorId} -> ${targetName}${lockedLabel}${exits}`;
      btn.disabled = locked.has(doorId) || state.agent.status !== "active";
      btn.onclick = () => doMove(target);
      container.appendChild(btn);
    });
  }

  function renderLastOp() {
    const node = document.getElementById("lastOp");
    node.textContent = state.lastOp ? JSON.stringify(state.lastOp, null, 2) : "No moves yet.";
  }

  function getAgentId() {
    return document.getElementById("agentId").value.trim() || "tester";
  }

  async function doMove(roomId) {
    const agentId = getAgentId();
    const res = await fetch(`/rooms/${encodeURIComponent(roomId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: agentId }),
    });
    const data = await res.json();
    state.lastOp = data;
    renderLastOp();
    await refreshAll();
  }

  async function registerAgent() {
    const agentId = getAgentId();
    await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agentId }),
    });
    await refreshAll();
  }

  async function resetMaze() {
    await fetch("/reset", { method: "POST" });
    state.lastOp = null;
    renderLastOp();
    await refreshAll();
  }

  async function refreshAll() {
    await fetchMaze();
    await fetchAgent();
  }

  document.getElementById("registerBtn").onclick = registerAgent;
  document.getElementById("resetBtn").onclick = resetMaze;

  refreshAll();
  setInterval(refreshAll, 2000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)
