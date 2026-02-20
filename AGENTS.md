# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python 3.12 exploration-guidance demo with multiple runnable components:
- Top-level apps and models: `simple_app.py`, `maze_app.py`, `tractor_app.py`, plus corresponding `*_model.py` files.
- Guidance and evaluation logic: `exploration_guidance_general.py`, `evaluation.py`, `exploration_cmd.py`.
- MCP server and HTTP tooling: `exploration_mcp/`.
- Agent implementations: `llm_agent/` and `bdi_agent/`.
- Ontologies (Turtle files): `ontologies/`.
- Integration script: `test_guidance.py`.

Keep new domain-specific modules close to related app/model files unless they are reusable across domains (then place them under `llm_agent/` or `exploration_mcp/`).

## Build, Test, and Development Commands
- `uv sync --dev`: install runtime and dev dependencies from `pyproject.toml`.
- `uv run ruff check .`: run lint checks (`E`, `F`) with line length 100.
- `uv run pyright`: run static type checking.
- `uv run python maze_app.py`: run the maze Flask app locally (default port in file).
- `uv run python exploration_cmd.py bob`: run the exploration CLI for an agent.
- `bash exploration.sh`: open the interactive HTTP/guidance shell.
- `uv run python test_guidance.py`: run current integration-style test flow.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Add type hints for new/modified public functions and data-flow-heavy helpers.
- Keep Flask route handlers thin; move reusable logic into helper functions.

## Testing Guidelines
There is no full pytest suite yet; `test_guidance.py` is an integration runner that starts services and validates guidance flow. For new behavior:
- Add targeted tests as `test_*.py` files (prefer deterministic, small-scope cases).
- Validate both API responses and guidance metadata where applicable.
- Run lint and type checks before opening a PR.

## Commit & Pull Request Guidelines
Recent history mixes short imperative commits (`Update maze app`) and Conventional Commit style (`feat: ...`, `fix(mcp): ...`). Prefer Conventional Commits for new work.

For each PR include:
- A concise summary of behavior changes.
- Reproduction/verification steps (exact commands run).
- Linked issue(s), if available.
- Request/response examples when API behavior changes.

## Security & Configuration Tips
- Do not commit secrets. Treat `API_KEY.txt` as local-only; prefer environment variables for credentials.
- Default localhost ports are hardcoded in several scripts; document any new port or URL assumptions in code comments and PR notes.
