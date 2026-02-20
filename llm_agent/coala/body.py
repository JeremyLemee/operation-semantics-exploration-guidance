from threading import Thread
from typing import Optional

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from llm_agent.coala.sensor import Sensor


class Body(Sensor):
    """
    Sensor implementation backed by a lightweight Flask REST API.

    Clients can submit percepts via POST /percepts with a JSON payload of the
    form {"agent_id": "...", "percept": "..."} and the percept will be queued
    for later aggregation via gather().
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        super().__init__()
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._server = None
        self._thread: Optional[Thread] = None
        self._setup_routes()

    def _setup_routes(self):
        body = self

        @self.app.post("/percepts")
        def add_percept_route():
            data = request.get_json(silent=True) or {}
            agent_id = data.get("agent_id")
            percept = data.get("percept")

            if not isinstance(percept, str) or not percept.strip():
                return jsonify(
                    {"error": "Field 'percept' must be a non-empty string"}
                ), 400

            formatted = body._format_percept(agent_id, percept)
            body.add_percept(formatted)
            return jsonify({"status": "queued"}), 201

    def _format_percept(self, agent_id: Optional[str], percept: str) -> str:
        """Store both agent and message while keeping Sensor API text-based."""
        if agent_id and agent_id.strip():
            return f"[agent:{agent_id.strip()}] {percept}"
        return percept

    def start(self):
        """Start the Flask server in a background thread if not already running."""
        if self._thread and self._thread.is_alive():
            return

        self._server = make_server(self.host, self.port, self.app)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the Flask server and wait for the background thread to exit."""
        if not self._thread:
            return

        self._server.shutdown()
        self._thread.join()
        self._thread = None
        self._server = None
