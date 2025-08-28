def run_multi(args):
    """
    Run multiple Snake games in parallel, all controlled by the NN.
    - Vectorized action selection via policy.act_batch
    - Optional tiled visualization
    - Supports continuous reset of finished envs until episodes_per_env reached
    """
    import math
    import pygame as pg
    from core.snake_rules import Rules, Config
    from core.snake_env import SnakeEnv
    from viz.renderer import Renderer
    from policy.policy import TorchPolicy
    import numpy as np
    import torch

    # ---- config ----
    n_envs            = getattr(args, "n_envs", 8)
    episodes_per_env  = getattr(args, "episodes_per_env", 3)
    obs_mode          = "flat"  # change to "ch3" if your policy expects that
    grid_w, grid_h    = args.grid_w, args.grid_h
    fps               = args.fps
    render            = getattr(args, "render", False)
    cell_px           = getattr(args, "cell_px", 20)

    # ---- build envs ----
    cfg = Config(grid_w=grid_w, grid_h=grid_h, relative_actions=True)
    envs   = [SnakeEnv(Rules(cfg), obs_mode=obs_mode) for _ in range(n_envs)]
    obss   = [env.reset(seed=(args.seed or 0) + i) for i, env in enumerate(envs)]
    dones  = np.zeros(n_envs, dtype=bool)
    ep_left = np.full(n_envs, episodes_per_env, dtype=int)
    returns = np.zeros(n_envs, dtype=np.float32)
    scores  = np.zeros(n_envs, dtype=int)

    # ---- policy (simple tiny MLP for demo; swap for your real model) ----
    flat_dim = int(np.prod(envs[0].observation_shape()))
    class TinyMLP(torch.nn.Module):
        def __init__(self, inp, out=3):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(inp, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, out)
            )
        def forward(self, x): return self.net(x)

    device = torch.device("cuda" if (torch.cuda.is_available() and not getattr(args, "cpu", False)) else "cpu")
    policy = TorchPolicy(TinyMLP(flat_dim), device=device, obs_mode=obs_mode)

    # ---- optional tiled renderer ----
    if render:
        pg.init()
        # choose a near-square tiling
        cols = math.ceil(math.sqrt(n_envs))
        rows = math.ceil(n_envs / cols)

        window_w = cols * grid_w * cell_px
        window_h = rows * grid_h * cell_px
        screen = pg.display.set_mode((window_w, window_h))
        pg.display.set_caption(f"Snake — {n_envs} envs")
        clock = pg.time.Clock()

        # Make one Renderer per env, each drawing into a subsurface tile
        renderers = []
        tiles = []
        for i in range(n_envs):
            r = Renderer(cell_px=cell_px)
            cx, cy = i % cols, i // cols
            rect = pg.Rect(cx * grid_w * cell_px, cy * grid_h * cell_px,
                           grid_w * cell_px, grid_h * cell_px)
            tile = screen.subsurface(rect)
            r.attach_surface(tile)
            renderers.append(r)
            tiles.append(tile)

    # ---- loop until every env completes its quota ----
    total_steps = 0
    while ep_left.sum() > 0:
        # handle quit when rendering
        if render:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    ep_left[:] = 0  # force exit
                elif e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                    ep_left[:] = 0

        # build batch from envs that are still active this step
        active_idx = np.where((ep_left > 0) & (~dones))[0]
        if len(active_idx) == 0:
            # everyone is either done for this step or done with episodes; handle resets
            fin_idx = np.where((ep_left > 0) & (dones))[0]
            for i in fin_idx:
                obss[i] = envs[i].reset(seed=(args.seed or 0) + total_steps + i)
                dones[i] = False
                returns[i] = 0.0
                scores[i]  = 0
            continue

        obs_batch = np.stack([obss[i] for i in active_idx], axis=0)
        acts = policy.act_batch(obs_batch)

        # step only active envs
        for j, i in enumerate(active_idx):
            step = envs[i].step(int(acts[j]))
            obss[i] = step.obs
            returns[i] += step.reward
            scores[i]   = step.info.get("score", scores[i])
            done_i = step.terminated or step.truncated

            # render this env's tile if needed
            if render:
                renderers[i].draw(step.info["snapshot"], score_text=True)

            if done_i:
                dones[i] = True
                ep_left[i] -= 1
                print(f"[env {i}] ep_done  rem={ep_left[i]}  return={returns[i]:.2f}  score={scores[i]}  reason={step.info.get('reason')}")
                # If more episodes remaining for that env, it will be reset on the next outer loop iteration

        total_steps += 1

        if render:
            pg.display.flip()
            clock.tick(fps)

    if render:
        pg.quit()
    print("All envs finished.")

