# snake/runners/run_py
from dataclasses import replace
from core.snake_rules import Rules
from core.snake_env import SnakeEnv
from viz.renderer_pygame import PygameRenderer
from viz.keyboard import Keyboard
from config import AppConfig
from rl.utils import dir_to_abs

def main(args):
    cfg = AppConfig()

    rules = Rules(cfg)
    env = SnakeEnv(cfg, rules)

    rend = PygameRenderer()
    rend.open(cfg)

    kbd = Keyboard(relative=cfg.relative_actions)

    obs = env.reset()
    done = False
    while not done:
        key = kbd.poll()
        if key == "quit":
            break

        if key is None:
            cur_dir = env.get_snapshot().dir
            action = dir_to_abs(cur_dir) if not cfg.relative_actions else 0
        else:
            action = key

        step = env.step(action)
        rend.draw(step.info["snapshot"])
        rend.tick(cfg.fps)
        done = step.terminated or step.truncated

    rend.close()

