import datetime as dt
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Tuple

import requests

EVALUATION_MAZE_FILE = os.environ.get("EVALUATION_MAZE_FILE", "maze.json")
os.environ["MAZE_FILE"] = EVALUATION_MAZE_FILE

from llm_agent.exploration_agent import run_agent_once  # noqa: E402
from maze_model import MazeModel  # noqa: E402


RUNS_PER_COMBINATION = 20
MAX_AGENT_CYCLES = 60
AGENT_ID = "bob"
GUIDANCE_ALLOWED_POLICIES = [
    "explorability"
]
ALLOWED_GOALS = ["exit1", "exit2", "exit3", "exit4"]
RESULTS_FILE = Path("results.txt")
SAVEPOINT_DIR = Path("evaluation_savepoints")
EVALUATION_DASHBOARD_HOST = "127.0.0.1"
EVALUATION_DASHBOARD_PORT = 8765

MAZE_RESET_URL = "http://127.0.0.1:5001/reset"
MAZE_STATUS_URL = "http://127.0.0.1:5001/status"
MAZE_HEALTH_URL = "http://127.0.0.1:5001/maze"
MCP_HEALTH_URL = "http://127.0.0.1:8100/mcp"
MCP_STATE_URL = "http://127.0.0.1:8101/state"
COALA_STOP_URL = "http://127.0.0.1:8001/stop"


class EvaluationState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Dict[str, str]] = []
        self._data: Dict = {
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "phase": "initializing",
            "dashboard_url": (
                f"http://{EVALUATION_DASHBOARD_HOST}:{EVALUATION_DASHBOARD_PORT}"
            ),
            "runs_per_combination": RUNS_PER_COMBINATION,
            "max_agent_cycles": MAX_AGENT_CYCLES,
            "agent_id": AGENT_ID,
            "guidance_policies": GUIDANCE_ALLOWED_POLICIES,
            "goals": [],
            "total_runs_planned": 0,
            "runs_completed": 0,
            "current_run": None,
            "latest_maze_status": None,
            "latest_agent_cycles": None,
            "latest_record": None,
            "failures": 0,
            "records": [],
            "events": [],
        }

    def update(self, **fields: object) -> None:
        with self._lock:
            self._data.update(fields)
            self._data["updated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()

    def add_event(self, message: str) -> None:
        event = {"time_utc": dt.datetime.now(dt.UTC).isoformat(), "message": message}
        with self._lock:
            self._events.append(event)
            self._events = self._events[-100:]
            self._data["events"] = list(self._events)
            self._data["updated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()

    def add_record(self, record: Dict) -> None:
        with self._lock:
            records = self._data.get("records", [])
            if not isinstance(records, list):
                records = []
            records.append(record)
            self._data["records"] = records[-100:]
            self._data["latest_record"] = record
            self._data["runs_completed"] = int(self._data.get("runs_completed", 0)) + 1
            if not record.get("success", False):
                self._data["failures"] = int(self._data.get("failures", 0)) + 1
            self._data["updated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()

    def snapshot(self) -> Dict:
        with self._lock:
            return json.loads(json.dumps(self._data))


EVAL_STATE = EvaluationState()


class EvaluationStopRequested(Exception):
    """Raised when the user requests stop from the dashboard."""


class EvaluationController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._start_request: Optional[Dict[str, str]] = None
        self._stop_requested = False

    def request_start_fresh(self) -> None:
        with self._cond:
            self._start_request = {"mode": "fresh"}
            self._cond.notify_all()

    def request_resume(self, savepoint_name: str) -> None:
        with self._cond:
            self._start_request = {"mode": "resume", "savepoint": savepoint_name}
            self._cond.notify_all()

    def wait_for_start(self) -> Dict[str, str]:
        with self._cond:
            while self._start_request is None:
                self._cond.wait(timeout=0.5)
            request = dict(self._start_request)
            self._start_request = None
            self._stop_requested = False
            return request

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested


EVAL_CONTROL = EvaluationController()


def _ensure_savepoint_dir() -> None:
    SAVEPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _savepoint_path(name: str) -> Path:
    candidate = (SAVEPOINT_DIR / name).resolve()
    if candidate.parent != SAVEPOINT_DIR.resolve():
        raise ValueError("Invalid savepoint name")
    return candidate


def _list_savepoints() -> List[Dict[str, object]]:
    _ensure_savepoint_dir()
    points: List[Dict[str, object]] = []
    for path in sorted(SAVEPOINT_DIR.glob("*.txt"), reverse=True):
        item: Dict[str, object] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at_utc": dt.datetime.fromtimestamp(
                path.stat().st_mtime, tz=dt.UTC
            ).isoformat(),
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                item["created_at_utc"] = payload.get("created_at_utc")
                item["runs_completed"] = payload.get("runs_completed")
                item["next_plan_index"] = payload.get("next_plan_index")
                item["next_goal_index"] = payload.get("next_goal_index")
                item["completed_goals"] = payload.get("completed_goals")
        except Exception:
            item["invalid"] = True
        points.append(item)
    return points


def _write_savepoint(payload: Dict[str, object]) -> Path:
    _ensure_savepoint_dir()
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = SAVEPOINT_DIR / f"savepoint_{timestamp}.txt"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_savepoint(savepoint_name: str) -> Dict[str, object]:
    path = _savepoint_path(savepoint_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Savepoint {savepoint_name} is not a JSON object")
    return payload


class _EvalDashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/state":
            state = EVAL_STATE.snapshot()
            state["savepoints"] = _list_savepoints()
            self._send_json(state)
            return
        if self.path == "/":
            self._send_html(self._render_dashboard())
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/control/stop":
            EVAL_CONTROL.request_stop()
            EVAL_STATE.add_event("Stop requested from dashboard")
            self._send_json({"ok": True, "message": "Stop requested"})
            return

        if self.path == "/control/start":
            phase = str(EVAL_STATE.snapshot().get("phase", ""))
            if phase != "awaiting_start":
                self._send_json(
                    {
                        "ok": False,
                        "error": (
                            "Start/resume is only allowed in awaiting_start phase, "
                            f"got {phase}"
                        ),
                    },
                    status=HTTPStatus.CONFLICT,
                )
                return
            payload = self._read_json_body()
            mode = str(payload.get("mode", ""))
            if mode == "fresh":
                EVAL_CONTROL.request_start_fresh()
                EVAL_STATE.add_event("Start requested from dashboard (fresh)")
                self._send_json({"ok": True, "message": "Fresh start requested"})
                return
            if mode == "resume":
                savepoint = str(payload.get("savepoint", ""))
                if not savepoint:
                    self._send_json(
                        {"ok": False, "error": "Missing savepoint"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    path = _savepoint_path(savepoint)
                    if not path.is_file():
                        raise RuntimeError("Savepoint file does not exist")
                except Exception:
                    self._send_json(
                        {"ok": False, "error": "Invalid savepoint path"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                EVAL_CONTROL.request_resume(savepoint)
                EVAL_STATE.add_event(
                    f"Resume requested from dashboard using {savepoint}"
                )
                self._send_json({"ok": True, "message": "Resume requested"})
                return

            self._send_json(
                {"ok": False, "error": "Invalid mode; use fresh or resume"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> Dict:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _render_dashboard() -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evaluation Dashboard</title>
  <style>
    :root { --bg:#f4f6f8; --card:#ffffff; --text:#16202a; --muted:#5e6b78;
            --accent:#0f6bff; --bad:#b42318; --ok:#027a48; --border:#d0d7de; }
    body { margin:0; padding:20px; font-family: Menlo, Consolas, monospace;
           background:var(--bg); color:var(--text); }
    .grid { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .card { background:var(--card); border:1px solid var(--border);
            border-radius:8px; padding:12px; }
    h1 { margin:0 0 12px 0; font-size:20px; }
    h2 { margin:0 0 8px 0; font-size:15px; color:var(--muted); }
    .kv { margin:3px 0; }
    .muted { color:var(--muted); }
    .ok { color:var(--ok); }
    .bad { color:var(--bad); }
    pre { margin:0; overflow:auto; white-space:pre-wrap; word-break:break-word;
          font-size:12px; line-height:1.35; }
  </style>
</head>
<body>
  <h1>Evaluation Dashboard</h1>
  <div class="card" style="margin-bottom:12px;">
    <h2>Controls</h2>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      <button onclick="startFresh()">Start Fresh</button>
      <select id="savepointSelect" style="min-width:260px;"></select>
      <button onclick="resumeSavepoint()">Resume Savepoint</button>
      <button onclick="requestStop()">Stop & Savepoint</button>
    </div>
    <div class="muted" id="controlMessage" style="margin-top:8px;"></div>
  </div>
  <div class="grid">
    <div class="card"><h2>Overview</h2><div id="overview"></div></div>
    <div class="card"><h2>Current Run</h2><div id="current"></div></div>
    <div class="card"><h2>Live Maze Status</h2><pre id="maze"></pre></div>
    <div class="card"><h2>Recent Events</h2><pre id="events"></pre></div>
    <div class="card"><h2>Latest Record</h2><pre id="latest"></pre></div>
    <div class="card"><h2>Recent Records</h2><pre id="records"></pre></div>
  </div>
  <script>
    function esc(v){
      return String(v ?? "n/a")
        .replace(/[<>&]/g, s => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[s]));
    }
    function pct(a,b){ if(!b) return "0.00%"; return ((a*100)/b).toFixed(2) + "%"; }
    async function postJson(url, payload){
      const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {})
      });
      return await res.json();
    }
    async function startFresh(){
      const out = await postJson('/control/start', {mode:'fresh'});
      document.getElementById('controlMessage').textContent = JSON.stringify(out);
    }
    async function resumeSavepoint(){
      const select = document.getElementById('savepointSelect');
      const savepoint = select.value;
      if(!savepoint){
        document.getElementById('controlMessage').textContent = 'Select a savepoint first';
        return;
      }
      const out = await postJson('/control/start', {mode:'resume', savepoint});
      document.getElementById('controlMessage').textContent = JSON.stringify(out);
    }
    async function requestStop(){
      const out = await postJson('/control/stop', {});
      document.getElementById('controlMessage').textContent = JSON.stringify(out);
    }
    async function refresh(){
      const res = await fetch('/state', {cache:'no-store'});
      const s = await res.json();
      const done = s.runs_completed || 0;
      const total = s.total_runs_planned || 0;
      const failures = s.failures || 0;
      const successes = Math.max(0, done - failures);
      document.getElementById('overview').innerHTML = `
        <div class="kv"><b>Phase:</b> ${esc(s.phase)}</div>
        <div class="kv"><b>Started:</b> ${esc(s.started_at_utc)}</div>
        <div class="kv"><b>Updated:</b> ${esc(s.updated_at_utc)}</div>
        <div class="kv"><b>Progress:</b> ${done}/${total} (${pct(done,total)})</div>
        <div class="kv"><b class="ok">Successes:</b> ${successes}
        | <b class="bad">Failures:</b> ${failures}</div>
        <div class="kv"><b>Goals:</b> ${esc((s.goals||[]).join(', '))}</div>
        <div class="kv"><b>Guidance Policies:</b> ${esc((s.guidance_policies||[]).join(', '))}</div>
        <div class="kv"><b>Savepoints:</b> ${esc((s.savepoints||[]).length)}</div>
      `;
      const select = document.getElementById('savepointSelect');
      const selected = select.value;
      const points = s.savepoints || [];
      const options = points.map(p =>
        `<option value="${esc(p.name)}">${esc(p.name)} (${esc(p.runs_completed)} runs)</option>`
      ).join('');
      select.innerHTML = '<option value="">Select savepoint</option>' +
        options;
      if(selected && points.some(p => p.name === selected)) {
        select.value = selected;
      }
      document.getElementById('current').innerHTML =
        `<pre>${esc(JSON.stringify(s.current_run, null, 2))}</pre>`;
      document.getElementById('maze').textContent = JSON.stringify({
        latest_maze_status: s.latest_maze_status,
        latest_agent_cycles: s.latest_agent_cycles
      }, null, 2);
      document.getElementById('events').textContent =
        JSON.stringify((s.events || []).slice(-20), null, 2);
      document.getElementById('latest').textContent = JSON.stringify(s.latest_record, null, 2);
      document.getElementById('records').textContent =
        JSON.stringify((s.records || []).slice(-10), null, 2);
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def _start_dashboard_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(
        (EVALUATION_DASHBOARD_HOST, EVALUATION_DASHBOARD_PORT), _EvalDashboardHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class ManagedProcess:
    def __init__(self, name: str, popen: subprocess.Popen, output_file):
        self.name = name
        self.popen = popen
        self.output_file = output_file

    def stop(self) -> None:
        try:
            if self.popen.poll() is None:
                self.popen.terminate()
                try:
                    self.popen.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.popen.kill()
                    self.popen.wait(timeout=5)
        finally:
            self.output_file.close()


def _agent_worker(
    goal: str, agent_id: str, max_cycles: int, result_queue: mp.Queue
) -> None:
    try:
        result = run_agent_once(
            goal=goal,
            agent_id=agent_id,
            max_cycles=max_cycles,
            enable_gui=True,
        )
        result_queue.put({"result": result})
    except Exception as exc:
        result_queue.put({"error": str(exc)})


class AgentRunner:
    def __init__(self, goal: str, agent_id: str, max_cycles: int):
        self.goal = goal
        self.agent_id = agent_id
        self.max_cycles = max_cycles
        self.result: Optional[Dict] = None
        self.error: Optional[Exception] = None
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_agent_worker,
            args=(goal, agent_id, max_cycles, self._queue),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def _collect_outcome(self) -> None:
        if self.result is not None or self.error is not None:
            return
        try:
            payload = self._queue.get(timeout=0.5)
        except Empty:
            return
        if not isinstance(payload, dict):
            self.error = RuntimeError("Invalid agent worker payload")
            return
        if "error" in payload:
            self.error = RuntimeError(str(payload.get("error", "unknown agent error")))
            return
        result = payload.get("result")
        if isinstance(result, dict):
            self.result = result
        else:
            self.error = RuntimeError("Missing agent result payload")

    def join(self, timeout: Optional[float] = None) -> None:
        self._process.join(timeout=timeout)
        if not self._process.is_alive():
            self._collect_outcome()

    def terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()

    def kill(self) -> None:
        if self._process.is_alive() and hasattr(self._process, "kill"):
            self._process.kill()


def _wait_for_http(url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1.5)
            if response.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _validate_evaluation_maze_service() -> None:
    response = requests.get(MAZE_HEALTH_URL, timeout=5)
    response.raise_for_status()
    payload = response.json()
    rooms = payload.get("rooms") if isinstance(payload, dict) else None
    if not isinstance(rooms, dict) or "room0" not in rooms or "start" in rooms:
        raise RuntimeError(
            f"The maze service at 127.0.0.1:5001 is not serving {EVALUATION_MAZE_FILE}. "
            "Stop the existing service or restart evaluation.py so it can start "
            f"test_guidance.py with MAZE_FILE={EVALUATION_MAZE_FILE}."
        )


def _ensure_service(
    name: str,
    health_url: str,
    cmd: List[str],
    cwd: Path,
    extra_health_urls: Optional[List[str]] = None,
) -> Optional[ManagedProcess]:
    extra_health_urls = extra_health_urls or []
    if _wait_for_http(health_url, timeout_s=1.0):
        for extra_url in extra_health_urls:
            if not _wait_for_http(extra_url, timeout_s=5.0):
                raise RuntimeError(
                    f"{name} is partially available: {health_url} responds, "
                    f"but required endpoint {extra_url} does not. Stop the existing "
                    "service on that port and restart evaluation.py."
                )
        if health_url == MAZE_HEALTH_URL:
            _validate_evaluation_maze_service()
        return None

    env = os.environ.copy()
    env["MAZE_FILE"] = EVALUATION_MAZE_FILE
    maze_file_path = cwd / EVALUATION_MAZE_FILE
    if not maze_file_path.is_file():
        raise RuntimeError(
            f"Evaluation maze file does not exist: {maze_file_path}. "
            "Set EVALUATION_MAZE_FILE to an existing maze JSON file or add the file."
        )
    output_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    popen = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=output_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ready = _wait_for_http(health_url, timeout_s=30.0)
    if ready:
        for extra_url in extra_health_urls:
            if not _wait_for_http(extra_url, timeout_s=30.0):
                ready = False
                health_url = extra_url
                break
    if not ready:
        if popen.poll() is None:
            popen.terminate()
            try:
                popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                popen.kill()
                popen.wait(timeout=5)
        output_file.seek(0)
        startup_output = output_file.read().strip()
        output_file.close()
        details = (
            f"\nStartup output:\n{startup_output}"
            if startup_output
            else "\nStartup output was empty."
        )
        raise RuntimeError(f"{name} did not become ready at {health_url}.{details}")
    try:
        if health_url == MAZE_HEALTH_URL:
            _validate_evaluation_maze_service()
    except Exception:
        ManagedProcess(name=name, popen=popen, output_file=output_file).stop()
        raise
    return ManagedProcess(name=name, popen=popen, output_file=output_file)


def _discover_exit_goals() -> List[str]:
    model = MazeModel("http://localhost:5001/exploration_guidance_info/")
    available_goals = sorted(
        [
            room_id
            for room_id, room in model._rooms.items()
            if room.get("type") == "exit"
        ]
    )
    selected_goals = [goal for goal in ALLOWED_GOALS if goal in available_goals]
    if not selected_goals:
        raise RuntimeError(
            "None of the required goals are available. "
            f"Required={ALLOWED_GOALS}, available={available_goals}"
        )
    return selected_goals


def _reset_maze() -> None:
    response = requests.post(MAZE_RESET_URL, timeout=5)
    response.raise_for_status()


def _set_mcp_guidance_policy(policy: str) -> None:
    if policy not in GUIDANCE_ALLOWED_POLICIES:
        raise ValueError(
            "Invalid MCP guidance policy: "
            f"{policy!r}. Expected one of {GUIDANCE_ALLOWED_POLICIES}."
        )
    response = requests.patch(
        MCP_STATE_URL,
        json={"tool_description_mode": policy},
        timeout=5,
    )
    response.raise_for_status()


def _read_status() -> Dict:
    response = requests.get(MAZE_STATUS_URL, params={"agent_id": AGENT_ID}, timeout=5)
    response.raise_for_status()
    return response.json()


def _read_status_with_retry(
    *, max_attempts: int = 5, delay_s: float = 0.25
) -> Optional[Dict]:
    last_error: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            return _read_status()
        except requests.HTTPError as exc:
            # `/status` returns 404 until the agent has been registered.
            # During startup/reset this is expected and should be retried quietly.
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                time.sleep(delay_s)
                continue
            last_error = exc
            time.sleep(delay_s)
        except Exception as exc:
            last_error = exc
            time.sleep(delay_s)
    if last_error is not None:
        EVAL_STATE.add_event(f"Transient status read failure: {last_error}")
    return None


def _shutdown_agent_via_http() -> bool:
    try:
        response = requests.post(COALA_STOP_URL, timeout=2.0)
        return response.status_code < 500
    except Exception:
        return False


def _is_terminal_status(status: str) -> bool:
    return status != "active"


def _normalize_record(rec: Dict) -> Dict:
    record = dict(rec)
    if "failure_reason" not in record:
        record["failure_reason"] = (
            None if bool(record.get("success", False)) else "unknown_failure_reason"
        )
    record.setdefault("final_status", None)
    record.setdefault("final_room", None)
    record.setdefault("move_steps", 0)
    return record


def _build_exception_record(
    *,
    goal: str,
    guidance_policy: str,
    run_idx: int,
    exc: Exception,
) -> Dict:
    return _normalize_record(
        {
            "goal": goal,
            "guidance_policy": guidance_policy,
            "run": run_idx,
            "success": False,
            "error": str(exc),
            "failure_reason": "exception",
            "final_status": "unknown",
            "final_room": None,
            "move_steps": 0,
            "cycles": 0,
            "timed_out": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )


def _run_single(
    goal: str,
    guidance_policy: str,
    run_idx: int,
) -> Dict:
    if EVAL_CONTROL.stop_requested():
        raise EvaluationStopRequested("Stop requested before run start")

    EVAL_STATE.update(
        phase="running",
        current_run={
            "goal": goal,
            "guidance_policy": guidance_policy,
            "run": run_idx,
            "status": "starting",
        },
        latest_maze_status=None,
        latest_agent_cycles=None,
    )
    EVAL_STATE.add_event(
        "Run "
        f"{run_idx} started for goal={goal}, guidance_policy={guidance_policy}"
    )
    _reset_maze()
    _set_mcp_guidance_policy(guidance_policy)

    runner = AgentRunner(
        goal=goal,
        agent_id=AGENT_ID,
        # Hard cap enforced inside run_agent_once; do not rely on Coala HTTP state.
        max_cycles=MAX_AGENT_CYCLES,
    )
    runner.start()

    failure_reason: Optional[str] = None
    stop_requested_during_run = False
    while runner.is_alive():
        if EVAL_CONTROL.stop_requested():
            stop_requested_during_run = True
            _shutdown_agent_via_http()
            EVAL_STATE.add_event(
                f"Run {run_idx} interrupted by dashboard stop request"
            )
            break

        status = _read_status_with_retry(max_attempts=3, delay_s=0.2)
        if status is None:
            time.sleep(0.25)
            continue
        EVAL_STATE.update(
            latest_maze_status=status,
            current_run={
                "goal": goal,
                "guidance_policy": guidance_policy,
                "run": run_idx,
                "status": "running",
                "maze_status": status,
                "agent_cycles": None,
            },
        )
        status_value = str(status.get("status", ""))
        if _is_terminal_status(status_value):
            _shutdown_agent_via_http()
            EVAL_STATE.add_event(
                f"Run {run_idx} terminated due to maze status={status_value}"
            )
            break

        time.sleep(0.25)

    runner.join(timeout=20.0)
    if runner.is_alive():
        _shutdown_agent_via_http()
        runner.join(timeout=5.0)
    if runner.is_alive():
        EVAL_STATE.add_event(
            f"Run {run_idx}: agent still alive after graceful stop; force-terminating process"
        )
        runner.terminate()
        runner.join(timeout=5.0)
    if runner.is_alive():
        EVAL_STATE.add_event(
            f"Run {run_idx}: process still alive after terminate; sending kill"
        )
        runner.kill()
        runner.join(timeout=2.0)
    if runner.is_alive():
        EVAL_STATE.add_event(
            f"Run {run_idx}: failed to stop agent process; continuing with failed run record"
        )
    if stop_requested_during_run:
        raise EvaluationStopRequested("Stop requested while run in progress")

    if runner.error is not None:
        raise RuntimeError(f"Agent run failed: {runner.error}") from runner.error

    run_metrics = runner.result or {}
    EVAL_STATE.update(latest_agent_cycles=run_metrics.get("cycles"))
    status = _read_status_with_retry(max_attempts=10, delay_s=0.25)
    if status is None:
        status = {"status": "unknown", "room": None, "budget": None}

    success = status.get("status") == "exited" and status.get("room") == goal
    if not success:
        status_value = str(status.get("status", ""))
        if status_value == "exited":
            failure_reason = "wrong_exit"
        elif status_value in {"bankrupt", "trapped", "no_affordance"}:
            failure_reason = status_value
        elif bool(run_metrics.get("timed_out", False)) or int(
            run_metrics.get("cycles", 0)
        ) >= MAX_AGENT_CYCLES:
            failure_reason = "cycle_limit_reached"
        elif status_value == "unknown":
            failure_reason = "maze_status_unavailable"
        elif status_value == "active":
            failure_reason = "agent_stopped_early"
        elif status_value:
            failure_reason = status_value
        else:
            failure_reason = "unknown_failure_reason"

    result = {
        "goal": goal,
        "guidance_policy": guidance_policy,
        "run": run_idx,
        "success": bool(success),
        "final_status": status.get("status"),
        "final_room": status.get("room"),
        "budget": status.get("budget"),
        "move_steps": run_metrics.get("move_steps", 0),
        "cycles": run_metrics.get("cycles", 0),
        "timed_out": run_metrics.get("timed_out", False),
        "failure_reason": failure_reason,
        "input_tokens": run_metrics.get("total_input_tokens", 0),
        "output_tokens": run_metrics.get("total_output_tokens", 0),
    }
    result = _normalize_record(result)
    EVAL_STATE.update(
        current_run={
            "goal": goal,
            "guidance_policy": guidance_policy,
            "run": run_idx,
            "status": "finished",
            "result": result,
        }
    )
    EVAL_STATE.add_event(
        (
            f"Run {run_idx} finished: success={result['success']}, "
            f"room={result['final_room']}, status={result['final_status']}, "
            f"failure_reason={result['failure_reason']}"
        )
    )
    return result


def _aggregate(records: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for rec in records:
        key = (
            rec["goal"],
            rec["guidance_policy"],
        )
        grouped.setdefault(key, []).append(rec)

    summary: List[Dict] = []
    for (goal, guidance_policy), group in sorted(grouped.items()):
        runs = len(group)
        successes = [r for r in group if r["success"]]
        failures = [r for r in group if not r.get("success", False)]
        success_count = len(successes)
        success_rate = (success_count / runs) if runs else 0.0
        failure_count = len(failures)

        successful_steps = [int(r["move_steps"]) for r in successes]
        avg_steps_success = (
            (sum(successful_steps) / len(successful_steps))
            if successful_steps
            else None
        )
        failure_reason_counts: Dict[str, int] = {}
        for rec in failures:
            reason_obj = rec.get("failure_reason")
            if isinstance(reason_obj, str) and reason_obj:
                reason = reason_obj
            elif rec.get("error") is not None:
                reason = "exception"
            else:
                reason = "unknown_failure_reason"
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

        failure_reason_stats: List[Dict[str, object]] = []
        for reason, count in sorted(
            failure_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            failure_reason_stats.append(
                {
                    "reason": reason,
                    "count": count,
                    "rate_of_runs": (count / runs) if runs else 0.0,
                    "rate_of_failures": (
                        (count / failure_count) if failure_count else 0.0
                    ),
                }
            )

        summary.append(
            {
                "goal": goal,
                "guidance_policy": guidance_policy,
                "runs": runs,
                "successes": success_count,
                "failures": failure_count,
                "success_rate": success_rate,
                "avg_steps_when_success": avg_steps_success,
                "failure_reason_stats": failure_reason_stats,
            }
        )

    return summary


def _write_results(records: List[Dict], summary: List[Dict]) -> None:
    lines: List[str] = []
    lines.append(f"Evaluation timestamp (UTC): {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"Runs per combination: {RUNS_PER_COMBINATION}")
    lines.append(f"Max agent cycles per run: {MAX_AGENT_CYCLES}")
    lines.append(f"Guidance policies: {GUIDANCE_ALLOWED_POLICIES}")
    lines.append("")

    lines.append("=== Aggregated Results ===")
    for row in summary:
        step_text = (
            f"{row['avg_steps_when_success']:.2f}"
            if row["avg_steps_when_success"] is not None
            else "N/A"
        )
        failure_reason_stats_text = json.dumps(
            row["failure_reason_stats"], sort_keys=True
        )
        lines.append(
            " | ".join(
                [
                    f"goal={row['goal']}",
                    f"guidance_policy={row['guidance_policy']}",
                    f"success_rate={row['success_rate']:.2%} ({row['successes']}/{row['runs']})",
                    f"failure_count={row['failures']}",
                    f"avg_steps_when_success={step_text}",
                    f"failure_reason_stats={failure_reason_stats_text}",
                ]
            )
        )

    lines.append("")
    lines.append("=== Per-Run Results ===")
    for rec in records:
        lines.append(json.dumps(rec, sort_keys=True))

    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_savepoint_payload(
    *,
    goals: List[str],
    guidance_policies: List[str],
    records: List[Dict],
    completed_goals: List[str],
    next_goal_index: int,
    next_plan_index: int,
) -> Dict[str, object]:
    summary = _aggregate(records)
    return {
        "version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "runs_per_combination": RUNS_PER_COMBINATION,
        "max_agent_cycles": MAX_AGENT_CYCLES,
        "agent_id": AGENT_ID,
        "guidance_policies": guidance_policies,
        "goals": goals,
        "total_runs_planned": (
            len(goals)
            * len(guidance_policies)
            * RUNS_PER_COMBINATION
        ),
        "next_goal_index": next_goal_index,
        "next_plan_index": next_plan_index,
        "completed_goals": completed_goals,
        "runs_completed": len(records),
        "records": records,
        "summary": summary,
    }


def _build_run_plan(
    goals: List[str], guidance_policies: List[str]
) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for goal in goals:
        for guidance_policy in guidance_policies:
            for run_idx in range(1, RUNS_PER_COMBINATION + 1):
                plan.append(
                    {
                        "goal": goal,
                        "guidance_policy": guidance_policy,
                        "run": run_idx,
                    }
                )
    return plan


def _run_goal(
    goal: str, guidance_policies: List[str]
) -> Tuple[List[Dict], bool]:
    goal_records: List[Dict] = []
    for guidance_policy in guidance_policies:
        for run_idx in range(1, RUNS_PER_COMBINATION + 1):
            try:
                rec = _run_single(
                    goal=goal,
                    guidance_policy=guidance_policy,
                    run_idx=run_idx,
                )
            except EvaluationStopRequested:
                return goal_records, True
            except Exception as exc:
                rec = _build_exception_record(
                    goal=goal,
                    guidance_policy=guidance_policy,
                    run_idx=run_idx,
                    exc=exc,
                )
                EVAL_STATE.add_event(
                    (
                        "Run "
                        f"{run_idx} failed with exception for "
                        f"goal={goal}, guidance_policy={guidance_policy}: {exc}"
                    )
                )
            normalized = _normalize_record(rec)
            goal_records.append(normalized)
            EVAL_STATE.add_record(normalized)
            if EVAL_CONTROL.stop_requested():
                return goal_records, True
    return goal_records, False


def _restore_progress_state(records: List[Dict]) -> None:
    failures = sum(1 for rec in records if not bool(rec.get("success", False)))
    EVAL_STATE.update(
        runs_completed=len(records),
        failures=failures,
        records=records[-100:],
        latest_record=records[-1] if records else None,
    )


def main() -> None:
    if not GUIDANCE_ALLOWED_POLICIES:
        raise RuntimeError("GUIDANCE_ALLOWED_POLICIES is empty.")

    project_root = Path(__file__).resolve().parent
    managed: List[ManagedProcess] = []
    dashboard_server = _start_dashboard_server()
    EVAL_STATE.add_event(
        f"Dashboard started at http://{EVALUATION_DASHBOARD_HOST}:{EVALUATION_DASHBOARD_PORT}"
    )

    try:
        EVAL_STATE.update(phase="starting_services")
        maze_proc = _ensure_service(
            name="maze+guidance server",
            health_url=MAZE_HEALTH_URL,
            cmd=[sys.executable, "test_guidance.py"],
            cwd=project_root,
        )
        if maze_proc is not None:
            managed.append(maze_proc)
            EVAL_STATE.add_event("Started managed process: maze+guidance server")
        else:
            EVAL_STATE.add_event("Reused existing maze+guidance server")

        mcp_proc = _ensure_service(
            name="exploration guidance MCP server",
            health_url=MCP_HEALTH_URL,
            extra_health_urls=[MCP_STATE_URL],
            cmd=[sys.executable, "exploration_mcp/exploration_guidance_mcp_server.py"],
            cwd=project_root,
        )
        if mcp_proc is not None:
            managed.append(mcp_proc)
            EVAL_STATE.add_event(
                "Started managed process: exploration guidance MCP server"
            )
        else:
            EVAL_STATE.add_event("Reused existing exploration guidance MCP server")

        goals = _discover_exit_goals()
        guidance_policies = list(GUIDANCE_ALLOWED_POLICIES)
        total_runs = (
            len(goals)
            * len(guidance_policies)
            * RUNS_PER_COMBINATION
        )
        runs_per_goal = (
            len(guidance_policies)
            * RUNS_PER_COMBINATION
        )
        run_plan = _build_run_plan(goals, guidance_policies)
        EVAL_STATE.update(
            phase="awaiting_start",
            goals=goals,
            guidance_policies=guidance_policies,
            total_runs_planned=total_runs,
            current_run=None,
            latest_maze_status=None,
            latest_agent_cycles=None,
            latest_record=None,
            runs_completed=0,
            failures=0,
            records=[],
        )
        EVAL_STATE.add_event(
            (
                f"Discovered goals={goals}, "
                f"guidance_policies={guidance_policies}, total_runs={total_runs}"
            )
        )
        EVAL_STATE.add_event(
            "Waiting for dashboard command: Start Fresh or Resume Savepoint"
        )

        committed_records: List[Dict] = []
        start_plan_index = 0
        while True:
            committed_records = []
            start_plan_index = 0
            start_request = EVAL_CONTROL.wait_for_start()
            mode = start_request.get("mode")
            if mode != "resume":
                EVAL_STATE.add_event("Starting fresh evaluation run")
                break

            try:
                savepoint_name = str(start_request.get("savepoint", ""))
                payload = _load_savepoint(savepoint_name)
                payload_goals = payload.get("goals")
                payload_guidance_policies = payload.get("guidance_policies")
                if payload_goals != goals:
                    raise RuntimeError(
                        "Savepoint goals do not match the current environment"
                    )
                if payload_guidance_policies != guidance_policies:
                    raise RuntimeError(
                        "Savepoint guidance_policies do not match current settings"
                    )
                records_obj = payload.get("records")
                if not isinstance(records_obj, list):
                    raise RuntimeError("Savepoint records are missing or invalid")
                committed_records = [
                    _normalize_record(rec) for rec in records_obj if isinstance(rec, dict)
                ]
                next_plan_index_obj = payload.get("next_plan_index")
                if next_plan_index_obj is None:
                    # Backward compatibility for older goal-level savepoints.
                    next_goal_index_obj = payload.get("next_goal_index", 0)
                    if isinstance(next_goal_index_obj, int):
                        start_plan_index = next_goal_index_obj * runs_per_goal
                    elif isinstance(next_goal_index_obj, str):
                        start_plan_index = int(next_goal_index_obj) * runs_per_goal
                    else:
                        raise RuntimeError(
                            "Savepoint next_goal_index has invalid type"
                        )
                elif isinstance(next_plan_index_obj, int):
                    start_plan_index = next_plan_index_obj
                elif isinstance(next_plan_index_obj, str):
                    start_plan_index = int(next_plan_index_obj)
                else:
                    raise RuntimeError("Savepoint next_plan_index has invalid type")
                if start_plan_index < 0 or start_plan_index > len(run_plan):
                    raise RuntimeError("Savepoint next_plan_index is out of range")
                _restore_progress_state(committed_records)
                EVAL_STATE.add_event(
                    "Loaded savepoint "
                    f"{savepoint_name} with {len(committed_records)} runs "
                    f"(next_plan_index={start_plan_index})"
                )
                break
            except Exception as exc:
                EVAL_STATE.add_event(f"Failed to load requested savepoint: {exc}")
                EVAL_STATE.update(phase="awaiting_start")

        next_plan_index = start_plan_index
        stop_requested = False
        for plan_index in range(start_plan_index, len(run_plan)):
            if EVAL_CONTROL.stop_requested():
                stop_requested = True
                next_plan_index = plan_index
                break

            item = run_plan[plan_index]
            goal = str(item["goal"])
            guidance_policy = str(item["guidance_policy"])
            run_idx = int(str(item["run"]))
            try:
                rec = _run_single(
                    goal=goal,
                    guidance_policy=guidance_policy,
                    run_idx=run_idx,
                )
            except EvaluationStopRequested:
                stop_requested = True
                next_plan_index = plan_index
                EVAL_STATE.add_event(
                    "Stop received while a run was in progress; savepoint keeps all "
                    "completed runs and resumes from the interrupted run."
                )
                break
            except Exception as exc:
                rec = _build_exception_record(
                    goal=goal,
                    guidance_policy=guidance_policy,
                    run_idx=run_idx,
                    exc=exc,
                )
                EVAL_STATE.add_event(
                    (
                        "Run "
                        f"{run_idx} failed with exception for goal={goal}, "
                        f"guidance_policy={guidance_policy}: {exc}"
                    )
                )

            normalized = _normalize_record(rec)
            committed_records.append(normalized)
            EVAL_STATE.add_record(normalized)
            next_plan_index = plan_index + 1

            # Auto-save after finishing all runs for the current goal block.
            goal_block_done = next_plan_index >= len(run_plan)
            if not goal_block_done:
                next_item = run_plan[next_plan_index]
                goal_block_done = str(next_item["goal"]) != goal
            if goal_block_done:
                next_goal_index = (
                    next_plan_index // runs_per_goal if runs_per_goal > 0 else len(goals)
                )
                completed_goals = goals[:next_goal_index]
                savepoint_payload = _build_savepoint_payload(
                    goals=goals,
                    guidance_policies=guidance_policies,
                    records=committed_records,
                    completed_goals=completed_goals,
                    next_goal_index=next_goal_index,
                    next_plan_index=next_plan_index,
                )
                savepoint_path = _write_savepoint(savepoint_payload)
                EVAL_STATE.add_event(
                    "Created auto-savepoint after completing "
                    f"goal={goal}: {savepoint_path.as_posix()}"
                )

            if EVAL_CONTROL.stop_requested():
                stop_requested = True
                break

        if stop_requested:
            next_goal_index = (
                next_plan_index // runs_per_goal if runs_per_goal > 0 else len(goals)
            )
            completed_goals = goals[:next_goal_index]
            EVAL_STATE.update(phase="creating_savepoint")
            savepoint_payload = _build_savepoint_payload(
                goals=goals,
                guidance_policies=guidance_policies,
                records=committed_records,
                completed_goals=completed_goals,
                next_goal_index=next_goal_index,
                next_plan_index=next_plan_index,
            )
            savepoint_path = _write_savepoint(savepoint_payload)
            EVAL_STATE.update(phase="stopped", current_run=None)
            EVAL_STATE.add_event(
                f"Created savepoint after stop request: {savepoint_path.as_posix()}"
            )
            return

        summary = _aggregate(committed_records)
        EVAL_STATE.update(phase="writing_results", summary=summary)
        _write_results(committed_records, summary)
        EVAL_STATE.update(phase="completed", current_run=None)
        EVAL_STATE.add_event(f"Wrote results to {RESULTS_FILE}")
    finally:
        EVAL_STATE.update(phase="stopping_services")
        for proc in reversed(managed):
            proc.stop()
            EVAL_STATE.add_event(f"Stopped managed process: {proc.name}")
        dashboard_server.shutdown()
        dashboard_server.server_close()


if __name__ == "__main__":
    main()
