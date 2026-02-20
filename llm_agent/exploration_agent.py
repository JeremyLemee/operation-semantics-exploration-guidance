import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path when running this script directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CONFIG_PATH = _ROOT / "config.json"

from config_loader import load_config
from llm_agent.web_agent import WebAgent

GOAL_PROMPT_TEMPLATE = """
You are an exploration agent in a maze scenario designed to demonstrate exploration guidance semantics and how an agent reasons about them with its own internal objective (unknown to the service providing guidance).

Critical first action: call the set_goal tool with parameter {goal}.

Critical second action: call the http_request tool with parameter "http://localhost:5001/status?agent_id={agent_id}".

Goal: reach {goal}.

You are: {agent_id}.

Maze semantics:
- The maze has many exits in different rooms. The agent's main goal is to reach the specified exit ({goal}).
- Agents move between rooms through doors. Passing through a door is typically an explorable operation, but some doors become locked after use. A locked door makes that operation non-explorable.
- There are trap rooms from which the agent can never escape. Moving to an exit or to a trap room are non-explorable operations with a danger cause of type maze:Final.
- Some rooms are owned (owner can be bob, alice, or someone else). Moving to an owned room is non-explorable with danger cause type maze:Fee because a fee is paid if the room does not belong to the agent.
- The agent starts with a budget. If the budget becomes negative after entering an owned room, the agent cannot move anymore and cannot achieve its goal.

Agents present:
- Two agents exist: bob and alice. Only {agent_id} is actually run.
- Rooms owned by the current agent can be used at no cost by that agent, even if the exploration guidance model marks the operation as non-explorable.
- Rooms owned by other agents are non-explorable for the model and also costly/risky.

Instruction: Use the exploration guidance information to decide when to act, while still optimizing for reaching {goal} under the above constraints.

Notes:
- If your budget decreases, update your budget using the corresponding tool. Otherwise, do not use that tool.
- Use the set_goal tool once at the beginning to register your goal, then the http_request tool once at the beginning, and the stop tool when your task is completed or when you can no longer move. Otherwise, only use the tools to move in the maze.

When you reach an exit, or a situation where you are no longer able to move, stop execution immediately by calling {{"tool": "stop"}}. Do nothing else. Do not call stop if you can still move in the maze.
"""

DEFAULT_GOAL = "exit1"
DEFAULT_AGENT_ID = "bob"


def build_goal_prompt(goal: str, agent_id: str) -> str:
    return GOAL_PROMPT_TEMPLATE.format(goal=goal, agent_id=agent_id)


def _count_move_steps(agent: WebAgent) -> int:
    gui_state = getattr(agent, "_gui_state", {}) or {}
    percepts = gui_state.get("percepts", []) if isinstance(gui_state, dict) else []
    move_steps = 0
    for entry in percepts:
        if not isinstance(entry, dict):
            continue
        percept = str(entry.get("percept", ""))
        if "Tool used: move_" in percept:
            move_steps += 1
    return move_steps


async def run_agent_once_async(
    *,
    goal: str = DEFAULT_GOAL,
    agent_id: str = DEFAULT_AGENT_ID,
    max_cycles: int = 100,
    enable_gui: bool = False,
) -> Dict[str, Any]:
    config = load_config(str(_CONFIG_PATH))
    eg_mcp = config["execution_guidance_mcp"]

    goal_prompt = build_goal_prompt(goal, agent_id)

    executor_agent = WebAgent(
        goal_prompt,
        mcp_servers=[{"name": "mcp_sem", "server_url": eg_mcp}],
        initial_memory={"current_state": "start"},
        enable_gui=enable_gui,
    )

    cycles = 0
    while not executor_agent.stop and cycles < max_cycles:
        await executor_agent.run_cycle()
        cycles += 1

    timed_out = not executor_agent.stop and cycles >= max_cycles
    if timed_out:
        executor_agent.stop = True

    return {
        "goal": goal,
        "agent_id": agent_id,
        "cycles": cycles,
        "timed_out": timed_out,
        "move_steps": _count_move_steps(executor_agent),
        "total_input_tokens": executor_agent.total_input_tokens,
        "total_output_tokens": executor_agent.total_output_tokens,
    }


def run_agent_once(
    *,
    goal: str = DEFAULT_GOAL,
    agent_id: str = DEFAULT_AGENT_ID,
    max_cycles: int = 100,
    enable_gui: bool = False,
) -> Dict[str, Any]:
    return asyncio.run(
        run_agent_once_async(
            goal=goal,
            agent_id=agent_id,
            max_cycles=max_cycles,
            enable_gui=enable_gui,
        )
    )


if __name__ == "__main__":
    run_agent_once(goal=DEFAULT_GOAL, max_cycles=100, enable_gui=True)
