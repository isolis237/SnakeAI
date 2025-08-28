# snake/runners/run_py
from core.snake_rules import Rules, Config
from core.snake_env import SnakeEnv
from viz.renderer import Renderer
from viz.keyboard import Keyboard

def main(grid_w=12, grid_h=12, fps=10):
    env = SnakeEnv(Rules(Config(grid_w=grid_w, grid_h=grid_h)))
    rend = Renderer(cell_px=24); rend.create_window(grid_w, grid_h, "Snake — Human")
    kbd = Keyboard(relative=True)

    obs = env.reset()
    done = False
    while not done:
        key = kbd.poll()
        if key == "quit": break
        action = key if isinstance(key, int) else 0  # straight if no key
        step = env.step(action)
        rend.draw(step.info["snapshot"])
        rend.tick(fps)
        done = step.terminated or step.truncated
    rend.close()

if __name__ == "__main__":
    main()
