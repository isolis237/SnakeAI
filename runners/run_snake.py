# snake/runners/run_py
from core.snake_rules import Rules, Config
from core.snake_env import SnakeEnv
from viz.renderer import Renderer
from viz.keyboard import Keyboard

def main(args):
    RELATIVE_ACTIONS=False
    DIRS = [(1,0),(0,1),(-1,0),(0,-1)]  # Right, Down, Left, Up
    def dir_to_abs(cur_dir):
        return DIRS.index(cur_dir)

    env = SnakeEnv(Rules(Config(grid_w=args.grid_w, grid_h=args.grid_h, relative_actions=RELATIVE_ACTIONS)))
    rend = Renderer(cell_px=args.cell_px); rend.create_window(args.grid_w, args.grid_h, "Snake — Human")
    kbd = Keyboard(relative=RELATIVE_ACTIONS)

    obs = env.reset()
    done = False
    while not done:
        key = kbd.poll()
        if key == "quit": break
        if key is None:
            # No key: continue current direction
            cur_dir = env.get_snapshot().dir
            action = dir_to_abs(cur_dir) if not env.rules.cfg.relative_actions else 0
        else:
            action = key

        step = env.step(action)
        rend.draw(step.info["snapshot"])
        rend.tick(args.fps)
        done = step.terminated or step.truncated
    rend.close()

if __name__ == "__main__":
    main()
