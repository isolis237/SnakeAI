# snake/runners/run_snake.py
from dataclasses import replace
from core.snake_rules import Rules, Config
from core.snake_env import SnakeEnv
from viz.renderer_pygame import PygameRenderer
from config import AppConfig
from viz.keyboard import Keyboard

def main(cfg: AppConfig):
    RELATIVE_ACTIONS = False
    DIRS = [(1,0),(0,1),(-1,0),(0,-1)]
    def dir_to_abs(cur_dir): return DIRS.index(cur_dir)

    env = SnakeEnv(Rules(Config(
        grid_w=cfg.env.grid_w,
        grid_h=cfg.env.grid_h,
        relative_actions=RELATIVE_ACTIONS,
        seed=cfg.dqn.seed,
    )))
    rend = PygameRenderer()

    render_cfg = replace(cfg.render, title="Snake — Human")

    rend.open(
        grid_w=cfg.env.grid_w,
        grid_h=cfg.env.grid_h,
        cfg=render_cfg,
    )
    kbd = Keyboard(relative=RELATIVE_ACTIONS)

    obs = env.reset()
    done = False
    while not done:
        key = kbd.poll()
        if key == "quit":
            break

        if key is None:
            cur_dir = env.get_snapshot().dir
            action = dir_to_abs(cur_dir) if not env.rules.cfg.relative_actions else 0
        else:
            action = key

        step = env.step(action)
        rend.draw(step.info["snapshot"])
        rend.tick(cfg.runner.fps)
        done = step.terminated or step.truncated

    rend.close()

if __name__ == "__main__":
    # This is for running the file directly, which is not the primary use case anymore.
    # For now we can leave it as is. A proper solution would be to have a default AppConfig.
    pass
