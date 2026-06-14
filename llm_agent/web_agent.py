from typing import Optional
from pathlib import Path
from langchain_core.language_models import LLM

from llm_agent.coala.coala import Coala
from llm import load_llm


from config_loader import load_config

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config.json"


class WebAgent(Coala):
    """A specialized agent based on Coala for web interactions."""

    def __init__(
        self,
        initial_prompt,
        llm: Optional[LLM] = None,  # qwen3:30b #gpt-4.1-nano
        tools=None,
        mcp_servers=None,
        initial_memory: Optional[dict] = None,
        enable_gui: bool = False,
        gui_host: str = "127.0.0.1",
        gui_port: int = 8001,
    ):

        if llm is None:
            cfg = load_config(str(_CONFIG_PATH))
            llm = load_llm(
                cfg["llm_agent"]["provider"],
                cfg["llm_agent"]["model"],
                reasoning=cfg["llm_agent"].get("reasoning"),
                thinking=cfg["llm_agent"].get("thinking"),
            )

        # Initialize the base Coala agent with our specific configuration
        super().__init__(
            llm=llm,
            tools=tools,
            initial_prompt=initial_prompt,
            initial_memory=initial_memory or {},
            mcp_servers=mcp_servers,
            enable_gui=enable_gui,
            gui_host=gui_host,
            gui_port=gui_port,
        )
