import re
import time
import threading
from typing import Any, Dict, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from llm_agent.coala.pm import ProceduralMemory
from llm_agent.coala.body import Body
from llm_agent.coala.sensor import Sensor

import json
from flask import Flask, jsonify, Response


class _ChatMemory:
    def __init__(self, k: int):
        self.k = k
        self.messages: list[BaseMessage] = []

    def _trim(self) -> None:
        if self.k is not None and len(self.messages) > self.k:
            self.messages = self.messages[-self.k :]

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content=content))

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content=content))

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self._trim()


class _WindowedChatMemory:
    def __init__(self, memory_key: str = "chat_history", k: int = 10):
        self.memory_key = memory_key
        self.chat_memory = _ChatMemory(k)

    def load_memory_variables(self, inputs):
        return {self.memory_key: list(self.chat_memory.messages)}


class _CoalaState(TypedDict, total=False):
    observations: str
    thought: Any
    decision: str


class CoalaLangGraph:
    def __init__(
        self,
        llm,
        tools=None,
        initial_prompt=None,
        initial_memory=None,
        body=None,
        mcp_servers=None,
        enable_gui: bool = False,
        gui_host: str = "127.0.0.1",
        gui_port: int = 8001,
    ):
        if tools is None:
            tools = []
        if initial_prompt is None:
            initial_prompt = (
                "You are an intelligent agent that can use tools to accomplish tasks."
            )
        if initial_memory is None:
            initial_memory = {}
        self.llm = llm
        self.initial_prompt = initial_prompt
        self.working_memory = _WindowedChatMemory(memory_key="chat_history", k=10)
        self.procedural_memory = ProceduralMemory(llm)
        for t in tools:
            self.procedural_memory.add_tool(t)
        if mcp_servers:
            for server in mcp_servers:
                if isinstance(server, dict):
                    self.procedural_memory.register_mcp_server(**server)
                elif isinstance(server, tuple) and len(server) >= 2:
                    name, server_url = server[0], server[1]
                    self.procedural_memory.register_mcp_server(
                        name=name, server_url=server_url
                    )
                else:
                    raise ValueError(
                        "MCP server entries must be dicts with registration args or (name, server_url) tuples."
                    )
        self.sensor = Sensor()
        if body is not None:
            self.sensor = Body()
        self.working_memory.chat_memory.add_ai_message(self.initial_prompt)
        self.data = initial_memory
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cycle_input_tokens = 0
        self.cycle_output_tokens = 0
        self.stop = False
        self.start_time = time.time()
        self.enable_gui = enable_gui
        self.gui_host = gui_host
        self.gui_port = gui_port
        self._gui_thread: Optional[threading.Thread] = None
        self._gui_lock = threading.Lock()
        self._gui_state: Dict[str, Any] = {
            "states": [],
            "percepts": [],
            "decisions": [],
            "current_memory": None,
            "memory_history": [],
        }
        self._last_state = None
        self._capture_state_from_memory(initial=True)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_CoalaState)
        graph.add_node("sync_tools", self._node_sync_tools)
        graph.add_node("observe", self._node_observe)
        graph.add_node("think", self._node_think)
        graph.add_node("decide", self._node_decide)
        graph.add_node("execute", self._node_execute)
        graph.set_entry_point("sync_tools")
        graph.add_edge("sync_tools", "observe")
        graph.add_edge("observe", "think")
        graph.add_edge("think", "decide")
        graph.add_edge("decide", "execute")
        graph.add_edge("execute", END)
        return graph.compile()

    async def _node_sync_tools(self, state: _CoalaState) -> Dict[str, Any]:
        await self.procedural_memory.sync_mcp_tools()
        return {}

    def _node_observe(self, state: _CoalaState) -> Dict[str, Any]:
        observations = self.retrieve_observations()
        self.process_observations(observations)
        self.retrieve_episodic_memory(query=observations)
        self.retrieve_procedural_memory(query=observations)
        return {"observations": observations}

    def _node_think(self, state: _CoalaState) -> Dict[str, Any]:
        thought = self.think()
        return {"thought": thought}

    def _node_decide(self, state: _CoalaState) -> Dict[str, Any]:
        thought = state.get("thought")
        if thought is None:
            thought = self.think()
        decision = self.decide(thought)
        return {"decision": decision}

    async def _node_execute(self, state: _CoalaState) -> Dict[str, Any]:
        decision = state.get("decision") or "{}"
        await self.execute_decision(decision)
        self._capture_state_from_memory()
        self.clean_memory()
        return {}

    def retrieve_observations(self):
        return self.sensor.gather()

    def process_observations(self, observations):
        self.working_memory.chat_memory.add_user_message(observations)
        if observations:
            self._record_percept(observations, source="observation")

    def retrieve_episodic_memory(self, query):
        results = []
        for res in results:
            self.working_memory.chat_memory.add_message(
                AIMessage(content=res.page_content)
            )

    def retrieve_procedural_memory(self, query):
        tools = self.procedural_memory.retrieve_tools(query)
        for tool in tools:
            print("tool type: ", type(tool))
            tool_description = self._format_tool_description(tool)
            self.working_memory.chat_memory.add_message(
                AIMessage(content=f"Tool available: {tool.name} - {tool_description}")
            )

    def extract_reply(self, text: str) -> str:
        if "</think>" not in text:
            return text.strip()

        match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def decide(self, thought):
        print("Deciding phase")
        memory_context = self.working_memory.load_memory_variables({})
        chat_history = memory_context.get("chat_history", "")
        available_tools = self.procedural_memory.retrieve_tools()
        tools_description = "\n".join(
            [f"- {self._format_tool_description(tool)}" for tool in available_tools]
        )
        decision_prompt = (
            f"Initial Goal and Context:\n{self.initial_prompt}\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            f"Conversation and Observation Context:\n{chat_history}\n\n"
            f"Last thought:\n{thought}\n"
            "Based on the above context, especially relying on the last thought, and available tools, what should I "
            "do next?\n"
            'If no clear action is needed, you can respond with a "noop" (no operation).\n'
            'For tool use, respond with a JSON object containing "tool" and "tool_input" fields.\n'
            'For noop, respond with: {"tool": "noop"}\n'
            'To stop the agent respond with: {"tool": "stop"}\n'
            'For updating the permanent JSON data, respond with {"tool": "permanent_memory", "field": "field_name", "value":"field_value"}\n'
            'To add a memory to the permanent episodic memory, respond with {"tool": "permanent_memory", "memory": "memory_content"}, where "memory_content" is the memory you want to store\n'
            'To add a memory to the RAG episodic memory, response with {"tool": "episodic_memory", "memory": "memory_content"}, where "memory_content" is the memory you want tp store\n'
            'To stop the agent respond with: {"tool": "stop"}\n'
            'For normal tool use, respond with: {"tool": "tool_name", "tool_input": "tool_input"} where "tool_input" is a JSON object with the names of parameters associated with their values. If the tool has no parameter, the tool_input is {}\n'
            "Important: Your response should be valid JSON and should be directly parsable into JSON\n"
        )
        decision = self.llm.invoke(decision_prompt)
        decision_str = ""
        if isinstance(decision, str):
            decision_str = decision
        elif isinstance(decision, AIMessage):
            print("is decision AI message")
            ai_message: AIMessage = decision
            if "input_tokens" in ai_message.usage_metadata:
                self.total_input_tokens += ai_message.usage_metadata["input_tokens"]
                self.cycle_input_tokens += ai_message.usage_metadata["input_tokens"]
            if "output_tokens" in ai_message.usage_metadata:
                self.total_output_tokens += ai_message.usage_metadata["output_tokens"]
                self.cycle_output_tokens += ai_message.usage_metadata["output_tokens"]
            decision_str = decision.text()
        print(f"Decision made: {decision_str}")
        self._record_decision(decision_str)
        return self.extract_reply(decision_str)

    def think(self):
        print("Thinking phase")
        memory_context = self.working_memory.load_memory_variables({})
        chat_history = memory_context.get("chat_history", "")
        available_tools = self.procedural_memory.retrieve_tools()
        tools_description = "\n".join(
            [f"- {self._format_tool_description(tool)}" for tool in available_tools]
        )
        print("Tool descriptions: ", tools_description)
        think_prompt = (
            f"Initial Goal and Context:\n{self.initial_prompt}\n\n"
            f"Available Tools:\n{tools_description}\n\n"
            f"Permanent Memory:\n{self.data}"
            f"Conversation and Observation Context:\n{chat_history}\n\n"
            "Based on the above context and available tools, what should I do next?\n"
            "You can either choose to use a tool, do a noop operation for no operation, or updating a field of the "
            "permanent memory with a given value\n"
            "Please rely on Chain of Thoughts to make your choice. \n"
        )

        thought = self.llm.invoke(think_prompt)
        thought_str = ""
        if isinstance(thought, str):
            thought_str = thought
        elif isinstance(thought, AIMessage):
            print("is think AI message")
            ai_message: AIMessage = thought
            if "input_tokens" in ai_message.usage_metadata:
                self.total_input_tokens += ai_message.usage_metadata["input_tokens"]
                self.cycle_input_tokens += ai_message.usage_metadata["input_tokens"]
            if "output_tokens" in ai_message.usage_metadata:
                self.total_output_tokens += ai_message.usage_metadata["output_tokens"]
                self.cycle_output_tokens += ai_message.usage_metadata["output_tokens"]
            thought_str = thought.text()

        self.working_memory.chat_memory.add_ai_message(thought_str)
        print(f"Thought made: {thought}")
        return thought

    async def execute_decision(self, d):
        try:
            decision = json.loads(d)
            tool_name = decision["tool"]
            print("tool name: ", tool_name)
            self._record_decision(decision)
            if tool_name == "noop":
                print("No operation needed at this time.")
                self.working_memory.chat_memory.add_ai_message(
                    "Decided to take no action at this time."
                )
                return
            if tool_name == "permanent_memory":
                print(
                    "Update permanent memory field: ",
                    decision["field"],
                    " with value: ",
                    decision["value"],
                )
                self.data[decision["field"]] = decision["value"]
                self.working_memory.chat_memory.add_ai_message(
                    f"Updated permanent memory field '{decision['field']}' to '{decision['value']}'"
                )
                return
            if tool_name == "episodic_memory":
                print(
                    "Update permanent memory field: ",
                    decision["field"],
                    " with value: ",
                    decision["value"],
                )
                self.data[decision["field"]] = decision["value"]
                self.working_memory.chat_memory.add_ai_message(
                    f"Updated permanent memory field '{decision['field']}' to '{decision['value']}'"
                )
                return
            if tool_name == "stop":
                self.stop = True
                print("Total input tokens: ", self.total_input_tokens)
                print("Total output tokens: ", self.total_output_tokens)
                stop_time = time.time()
                total_time = stop_time - self.start_time
                print("total time: ", total_time)
                return
            if tool_name == "remember":
                return
            print("before looking for tools")
            tool = self.procedural_memory.get_tool(tool_name)
            print("tool found")
            if tool:
                try:
                    result = None
                    tool_input = decision.get("tool_input", {})
                    if isinstance(tool_input, dict):
                        tool_input = self._normalize_tool_input(
                            tool_name, tool, tool_input
                        )
                        if d == "{}":
                            print("Invoke tool without params")
                            result = await tool.ainvoke()
                        else:
                            result = await tool.ainvoke(tool_input)
                        print(f"Executed {tool_name}, result: {result}")
                    else:
                        print("tool input could not be used.")
                    percept = (
                        "Tool used: "
                        + tool_name
                        + " Tool input: "
                        + str(decision["tool_input"])
                        + ". Tool result: "
                        + str(result)
                    )
                    print("new percept: " + percept)
                    self.sensor.add_percept(percept)
                    self._record_percept(percept, source="tool")
                    self.working_memory.chat_memory.add_ai_message(percept)
                except Exception as e:
                    print(f"Tool execution failed: {e}")
                    self.working_memory.chat_memory.add_ai_message(
                        f"Tool execution failed: {str(e)}"
                    )
            else:
                print(f"No tool named '{tool_name}' found. Executing default fallback.")
                self.working_memory.chat_memory.add_ai_message(
                    f"Could not find tool '{tool_name}'."
                )
        except Exception as e:
            print(f"No valid JSON for {d} with type: {type(d)}")
            self._record_decision(
                {"error": "invalid_json", "raw": d, "exception": str(e)}
            )

    def clean_memory(self):
        history = self.working_memory.chat_memory.messages
        if len(history) > 20:
            self.working_memory.chat_memory.messages = history[-10:]

    def register_mcp_server(
        self,
        name: str,
        *,
        server_url: str = None,
        command: str = None,
        args=None,
        env=None,
    ):
        if server_url is None and command is None:
            if name in self.procedural_memory.mcp_servers:
                return
        self.procedural_memory.register_mcp_server(
            name=name,
            server_url=server_url,
            command=command,
            args=args,
            env=env,
        )

    async def run_cycle(self):
        start_time = time.time()
        self.cycle_input_tokens = 0
        self.cycle_output_tokens = 0
        await self._graph.ainvoke({})
        print("cycle input tokens: ", self.cycle_input_tokens)
        print("cycle output tokens: ", self.cycle_output_tokens)
        end_time = time.time()
        cycle_time = end_time - start_time
        print("cycle time: ", cycle_time)

    async def start(self):
        self._ensure_gui()
        while not self.stop:
            await self.run_cycle()

    def _capture_state_from_memory(self, initial: bool = False):
        current_state = None
        current_memory = None
        if isinstance(self.data, dict):
            current_state = self.data.get("current_state")
            current_memory = dict(self.data)
        if current_state is not None and current_state != self._last_state:
            self._record_state(current_state)
            self._last_state = current_state
        with self._gui_lock:
            self._gui_state["current_memory"] = current_memory
            if current_memory is not None:
                self._gui_state["memory_history"].append(
                    {"timestamp": time.time(), "memory": current_memory}
                )
                if len(self._gui_state["memory_history"]) > 200:
                    self._gui_state["memory_history"] = self._gui_state[
                        "memory_history"
                    ][-200:]

    def _record_state(self, state: str):
        with self._gui_lock:
            self._gui_state["states"].append({"timestamp": time.time(), "state": state})

    def _record_percept(self, percept: str, source: str):
        with self._gui_lock:
            self._gui_state["percepts"].append(
                {"timestamp": time.time(), "source": source, "percept": percept}
            )

    def _record_decision(self, decision: Any):
        with self._gui_lock:
            self._gui_state["decisions"].append(
                {"timestamp": time.time(), "decision": decision}
            )

    def _normalize_tool_input(
        self, tool_name: str, tool: Any, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        if tool_name == "update_profile":
            if "nl_context" not in tool_input and "context" in tool_input:
                tool_input = dict(tool_input)
                tool_input["nl_context"] = tool_input.pop("context")
        required_fields = getattr(tool, "required_fields", []) or []
        if len(required_fields) == 1:
            required = required_fields[0]
            if required not in tool_input and "context" in tool_input:
                tool_input = dict(tool_input)
                tool_input[required] = tool_input.pop("context")
        return tool_input

    def _ensure_gui(self):
        if not self.enable_gui or self._gui_thread is not None:
            return
        app = Flask(__name__)

        @app.get("/")
        def index() -> Response:
            html = """
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Coala Agent Monitor</title>
                <style>
                  :root { color-scheme: light; }
                  body { font-family: "Georgia", "Times New Roman", serif; background: linear-gradient(120deg, #f5f1e8, #efe6d4); color: #1f1a12; margin: 0; }
                  header { padding: 24px 32px; border-bottom: 1px solid #d8cbb3; background: rgba(255,255,255,0.6); backdrop-filter: blur(8px); }
                  h1 { margin: 0 0 8px 0; font-size: 28px; letter-spacing: 0.5px; }
                  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; padding: 24px 32px 40px; }
                  .panel { background: rgba(255,255,255,0.75); border: 1px solid #d8cbb3; border-radius: 14px; padding: 16px; box-shadow: 0 10px 24px rgba(0,0,0,0.08); }
                  .panel h2 { margin: 0 0 10px; font-size: 18px; text-transform: uppercase; letter-spacing: 1px; }
                  .list { max-height: 320px; overflow: auto; padding-right: 8px; }
                  .item { padding: 10px; border-bottom: 1px dashed #d8cbb3; font-size: 14px; }
                  .item:last-child { border-bottom: none; }
                  .meta { font-size: 12px; opacity: 0.7; margin-bottom: 4px; }
                  .current { font-size: 20px; font-weight: 700; }
                </style>
              </head>
              <body>
                <header>
                  <h1>Coala Agent Monitor</h1>
                  <div class="current" id="current-state">Current memory: --</div>
                </header>
                <section class="grid">
                  <div class="panel">
                    <h2>States</h2>
                    <div class="list" id="states"></div>
                  </div>
                  <div class="panel">
                    <h2>Percepts</h2>
                    <div class="list" id="percepts"></div>
                  </div>
                  <div class="panel">
                    <h2>Decisions</h2>
                    <div class="list" id="decisions"></div>
                  </div>
                  <div class="panel">
                    <h2>Memory Evolution</h2>
                    <div class="list" id="memory-history"></div>
                  </div>
                </section>
                <script>
                  const fmt = (ts) => new Date(ts * 1000).toLocaleTimeString();
                  async function refresh() {
                    const res = await fetch('/api/state');
                    const data = await res.json();
                    const memoryText = data.current_memory ? JSON.stringify(data.current_memory) : '--';
                    document.getElementById('current-state').textContent = `Current memory: ${memoryText}`;
                    const states = data.states.slice().reverse().map(entry => `
                      <div class="item"><div class="meta">${fmt(entry.timestamp)}</div>${entry.state}</div>
                    `).join('');
                    document.getElementById('states').innerHTML = states || '<div class="item">No states yet.</div>';
                    const percepts = data.percepts.slice().reverse().map(entry => `
                      <div class="item"><div class="meta">${fmt(entry.timestamp)} · ${entry.source}</div>${entry.percept}</div>
                    `).join('');
                    document.getElementById('percepts').innerHTML = percepts || '<div class="item">No percepts yet.</div>';
                    const decisions = data.decisions.slice().reverse().map(entry => `
                      <div class="item"><div class="meta">${fmt(entry.timestamp)}</div><pre>${JSON.stringify(entry.decision, null, 2)}</pre></div>
                    `).join('');
                    document.getElementById('decisions').innerHTML = decisions || '<div class="item">No decisions yet.</div>';
                    const history = data.memory_history.slice().reverse().map(entry => `
                      <div class="item"><div class="meta">${fmt(entry.timestamp)}</div><pre>${JSON.stringify(entry.memory, null, 2)}</pre></div>
                    `).join('');
                    document.getElementById('memory-history').innerHTML = history || '<div class="item">No memory snapshots yet.</div>';
                  }
                  refresh();
                  setInterval(refresh, 1500);
                </script>
              </body>
            </html>
            """
            return Response(html, mimetype="text/html")

        @app.get("/api/state")
        def api_state():
            with self._gui_lock:
                return jsonify(self._gui_state)

        def _run():
            app.run(
                host=self.gui_host, port=self.gui_port, debug=False, use_reloader=False
            )

        self._gui_thread = threading.Thread(target=_run, daemon=True)
        self._gui_thread.start()

    def _format_tool_description(self, tool) -> str:
        base_desc = ""
        if hasattr(tool, "describe") and callable(getattr(tool, "describe")):
            base_desc = tool.describe() or ""
        else:
            base_desc = getattr(tool, "description", "") or ""

        schema = self._get_tool_input_schema(tool)
        if not isinstance(schema, dict):
            return f"{tool.name}: {base_desc}".strip(": ")

        params = self._extract_param_descriptions(schema)
        if not params:
            return f"{tool.name}: {base_desc}".strip(": ")

        param_lines = []
        for name, desc, required in params:
            required_tag = " (required)" if required else ""
            detail = f"{name}{required_tag}"
            if desc:
                detail = f"{detail} - {desc}"
            param_lines.append(detail)
        param_text = "; ".join(param_lines)
        if base_desc:
            return f"{tool.name}: {base_desc} | params: {param_text}"
        return f"{tool.name}: params: {param_text}"

    def _get_tool_input_schema(self, tool):
        schema = getattr(tool, "input_schema", None) or getattr(
            tool, "inputSchema", None
        )
        if schema is None:
            schema = getattr(tool, "_input_schema", None)
        if schema is not None:
            return schema
        server = getattr(tool, "server", None)
        if server is not None:
            tool_def = getattr(server, "tool_definitions", {}).get(
                getattr(tool, "name", "")
            )
            if tool_def is not None:
                return getattr(tool_def, "input_schema", None) or getattr(
                    tool_def, "inputSchema", None
                )
        return None

    def _extract_param_descriptions(
        self, schema: Dict[str, Any]
    ) -> list[tuple[str, str, bool]]:
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = (
            set(schema.get("required", [])) if isinstance(schema, dict) else set()
        )
        params = []
        if isinstance(properties, dict):
            for name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    params.append((name, "", name in required))
                    continue
                desc = (
                    prop_schema.get("description", "")
                    or prop_schema.get("title", "")
                    or ""
                )
                params.append((name, desc, name in required))
        return params
