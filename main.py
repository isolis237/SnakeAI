# main.py
import random
from wall import SnakeWall
from snake import SnakeGame

def play_manual():
    g = SnakeGame(fps=10)
    g.init("Snake - Manual")
    g.reset()
    while g.running and not g.done:
        _, _, done, _ = g.step(action=None)
        if done: break
    g.close()

def play_agent():
    g = SnakeGame(fps=8)
    g.init("Snake - Agent")
    g.reset()
    rng = random.Random(0)
    while g.running and not g.done:
        action = rng.choice([0, 1, 2])
        _, _, done, _ = g.step(action)
        if done: break
    g.close()

def run_multi():
    # grid_w/h -> discrete positions(steps) exist horizontally/vertically in each game
    # cell -> how many pixels in each grid square / position

    # max ~ SnakeWall(rows=105, cols=210, grid_w=12, grid_h=12, cell= 1, fps=12, seed=0, restart_finished=True)
    # comfortable 
    wall = SnakeWall(rows=4, cols=5, grid_w=12, grid_h=12, cell=22, fps=12, seed=0, restart_finished=True)
    wall.run()

if __name__ == "__main__":
    # play_manual()
    # play_agent()
    run_multi()
