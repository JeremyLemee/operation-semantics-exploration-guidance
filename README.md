# exploration-guidance-py
A Python project to implement exploration guidance. OpenAI Codex was used in the coding of the project.


## Ontologies

The proposed ontology for exploration for exploration guidance, that defines our operation semantics is available [`here`](ontologies/guidance.ttl). 

The ontology used to represent the maze environment is available [`here`](ontologies/maze.ttl). 

## Exploration Guidance

The file [`exploration_guidance_general.py`](exploration_guidance_general.py) provides the classes (Model, ExplorationGuidanceServer) used to define exploration guidance.

The file [`maze_app.py`](maze_app.py) defines the Maze HTTP server without exploration guidance (it needs to be wrapped within an ExplorationGuidanceServer and a model to do so). The model for the maze is provided in the file [`maze_model.py`](maze_model.py).

## Launching the project

You need to have [`uv`](https://docs.astral.sh/uv/) installed.

Set an OpenAI API key in [`API_KEY.txt`](API_KEY.txt) to use an OpenAI model. The model to use can be configured in [`config.json`](config.json). 

Define the structure of the maze in [`maze.json`](maze.json).

### Run the server

The server can be run with: 
```
uv run test_guidance.py
```

This opens a server at: http://localhost:5001/.

### Run the MCP server

The code of MCP server implementing exploration guidance is available [`here`](exploration_mcp/exploration_guidance_mcp_server.py). The server is running at http://localhost:8100/mcp. You can use this [`script`](exploration_mcp/mcp_http_cli.py) to interact with it.

### Run the agent

The code of the exploration agent is available [`here`](llm_agent/exploration_agent.py).

### Evaluation

The evaluation can be run with: 
```
uv run evaluation.py
```

This opens a server at: http://localhost:8765/, which can be used to start, pause, or restart an evaluation.

Before running the script, you can configure the guidance policies that will be applied in this list variable: GUIDANCE_ALLOWED_POLICIES. The possible values are: "all", "none", "outcome", "danger", "explorability", and "outcome_only"