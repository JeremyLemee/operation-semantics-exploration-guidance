import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path when running this script directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CONFIG_PATH = _ROOT / "config.json"

from config_loader import load_config  # noqa: E402
from llm_agent.web_agent import WebAgent  # noqa: E402

GOAL_PROMPT_TEMPLATE = (
    "\n"
    "You are an exploration agent in a maze scenario designed to demonstrate exploration "
    "guidance semantics and how an agent reasons about them with its own internal objective.\n"
    "\n"
    'Critical first action: call the http_post_request tool with URL '
    '"http://localhost:5001/register" and JSON body "{{\\"name\\": \\"{agent_id}\\"}}".\n'
    "\n"
    "Goal: reach room {goal}.\n"
    "\n"
    "You are: {agent_id}.\n"
    "\n"
    "Maze semantics:\n"
    "- Room names follow the form room{{i}}. Rooms do not have owner metadata and generally "
    "do not have special room types.\n"
    "- Agents must register before they can act. Registration gives the agent an initial "
    "budget of 10.\n"
    "- Agents move between rooms through doors. Passing through a door is usually "
    "explorable, but some doors become locked after use.\n"
    "- Guidance may include outcomes such as the destination room and potential reachable "
    "goal rooms. Use that information to plan toward {goal}.\n"
    "- If you reach a room with no available moves, you are stuck and should stop.\n"
    "\n"
    "Agents present:\n"
    "- Only {agent_id} is actually run.\n"
    "\n"
    "Instruction: Use the exploration guidance information to decide when to act, while "
    "still optimizing for reaching room {goal}.\n"
    "\n"
    "Notes:\n"
    "- Call http_post_request exactly once at the beginning to register "
    "- After that, only use maze operation tools and stop.\n"
    "\n"
    "When a response shows that your current room is {goal}, stop execution immediately "
    'by calling {{"tool": "stop"}}. Also stop when you can no longer move. Do nothing '
    "else after either condition is true.\n"
)

DEFAULT_GOAL = "room9"
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
