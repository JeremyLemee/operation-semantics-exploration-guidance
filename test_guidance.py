from exploration_guidance_general import ExplorationGuidance
from maze_app import app as maze_app
from maze_model import MazeModel

maze_model = MazeModel("http://localhost:5001/exploration_guidance_info/")
guidance_app = ExplorationGuidance(maze_model, maze_app)


if __name__ == "__main__":
    guidance_app.run(port=5001, use_reloader=False)
