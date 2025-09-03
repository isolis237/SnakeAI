# main.py
import argparse
import numpy as np
import torch
from collections import deque

# ---- snake side ----
from core.snake_rules import Rules, Config
from core.snake_env import SnakeEnv
from viz.renderer import Renderer
from viz.keyboard import Keyboard

# ---- policy side ----
from policy.policy import TorchPolicy


from runners.run_snake import main as snake

def run_snake(args):
    snake(args)


def run_policy(args):
    """Run NN only (feed it random observations, see outputs)."""
    flat_dim = args.grid_w * args.grid_h * 3
    model = TinyMLP(flat_dim)
    pol = TorchPolicy(model, device=torch.device("cpu"), obs_mode="flat")

    rng = np.random.default_rng(args.seed)
    cnt = np.zeros(3, dtype=int)
    for _ in range(100):
        obs = rng.random(flat_dim, dtype=np.float32)
        a = pol.act(obs)
        cnt[a] += 1
    print("NN-only mode action histogram:", cnt)


def run_loop(args):
    """
    Single env, single OS window split into:
      - LEFT: game board
      - RIGHT: NN viewer
    Env stays on CPU; NN runs on GPU unless --cpu.
    """
    import numpy as np, pygame as pg, torch, random, copy, math
    from collections import deque
    from core.snake_rules import Rules, Config
    from core.snake_env import SnakeEnv
    from viz.renderer import Renderer          # must NOT auto-flip when attached_surface is used
    from viz.net_viewer import NetViewer       # must NOT auto-flip when attach_surface is used
    from viz.nn_viz import attach_activation_hooks
    from policy.policy import TorchPolicy

    # ---- tiny replay buffer ----
    class Replay:
        def __init__(self, cap=100_000):
            self.buf = deque(maxlen=cap)
        def push(self, s, a, r, s2, d):
            self.buf.append((s, a, r, s2, d))
        def sample(self, n):
            batch = random.sample(self.buf, n)
            s, a, r, s2, d = zip(*batch)
            return (np.stack(s), np.array(a),
                    np.array(r, dtype=np.float32),
                    np.stack(s2), np.array(d, dtype=np.float32))
        def __len__(self):
            return len(self.buf)

    # ---- tiny demo model (replace with yours) ----
    class TinyMLP(torch.nn.Module):
        def __init__(self, inp, out=3, hidden=1024):
            super().__init__()
            self.net = torch.nn.Sequential(
            torch.nn.Linear(inp, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out),
        )
        def forward(self, x):
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self.net(x)

    # ---- build env (CPU) ----
    cfg = Config(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        relative_actions=getattr(args, "relative_actions", True),
    )
    env = SnakeEnv(
    Rules(cfg),
    obs_mode="flat",          # keep flat for your MLP
    include_walls=True,
    include_dir=True,
)
    flat_dim = int(np.prod(env.observation_shape()))
    n_actions = env.action_space_n()  # 3 if relative, 4 if absolute

    # ---- model on GPU if available ----
    device = torch.device("cuda" if (torch.cuda.is_available() and not getattr(args, "cpu", False)) else "cpu")
    model = TinyMLP(flat_dim, out=n_actions).to(device)
    target = copy.deepcopy(model).to(device).eval()  # target net stays in eval mode

    import torch.nn as nn
    def weight_l2(m): return sum((p.detach()**2).sum().item() for p in m.parameters())**0.5
    w0 = weight_l2(model)

    # Important: for a Sequential under attribute `net`, hook names are "net.0","net.2",...
    acts_dict = attach_activation_hooks(model, target_layers=("net.0", "net.2"))

    # ---- create the single window and split into panes ----
    cell = getattr(args, "cell_px", 24)
    left_w  = cfg.grid_w * cell
    left_h  = cfg.grid_h * cell
    right_w = max(360, left_w // 1)
    right_h = left_h

    # pg.init()
    # screen = pg.display.set_mode((left_w + right_w, left_h))
    # pg.display.set_caption("Snake — Game + NN")
    # clock = pg.time.Clock()

    # # LEFT tile: game
    # game_tile = screen.subsurface(pg.Rect(0, 0, left_w, left_h))
    # renderer = Renderer(cell_px=cell)
    # renderer.attach_surface(game_tile)

    # # RIGHT tile: NN viewer
    # nn_tile = screen.subsurface(pg.Rect(left_w, 0, right_w, right_h))
    # net_viewer = NetViewer(
    #     layer_sizes=[min(flat_dim, 64), 32, n_actions],
    #     layer_keys=[None, "net.0", "net.2"],
    #     title="NN Viewer (pane)",
    #     w=right_w, h=right_h,
    #     surface=nn_tile,
    # )

    # Policy sends activations/logits to the viewer each forward pass
    policy = TorchPolicy(
        model=model,
        device=device,
        obs_mode="flat",
        #vis_callback=lambda **kw: net_viewer.push(**kw),
    )
    policy._acts = acts_dict  # allow hooks to populate activations

    # ---- DQN training bits ----
    replay = Replay(cap=200_000)                # bigger buffer = more diverse batches
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    gamma = 0.99

    BATCH        = 128
    WARMUP       = 400                       # collect experience before learning
    OPT_EVERY    = 8                            # less frequent updates = stabler
    TARGET_EVERY = 0                            # disable hard copy if you use Polyak (below)

    # Epsilon schedule: explore longer; keep some randomness
    eps_start, eps_final, eps_decay = 1.0, 0.05, 15_000
    def epsilon(step):
        t = min(1.0, step / float(eps_decay))
        return eps_start + (eps_final - eps_start) * t

    # OPTIONAL: Polyak (soft) target updates (recommended instead of hard copies)
    tau = 0.03
    def soft_update(target, online):
        with torch.no_grad():
            for p_t, p in zip(target.parameters(), online.parameters()):
                p_t.data.mul_(1 - tau).add_(tau * p.data)

    global_step = 0
    loss_avg = 0.0

    def optimize():
        nonlocal loss_avg
        model.train()
        s, a, r, s2, d = replay.sample(BATCH)

        s_t  = torch.from_numpy(s).float().to(device)
        a_t  = torch.from_numpy(a).long().to(device)
        r_t  = torch.from_numpy(r).to(device)
        s2_t = torch.from_numpy(s2).float().to(device)
        d_t  = torch.from_numpy(d).to(device)

        # Q(s,a)
        q = model(s_t)                                # (B, A)
        q_sa = q.gather(1, a_t.view(-1, 1)).squeeze(1)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q2_online = model(s2_t)                    # (B, A)
            a2_online = q2_online.argmax(dim=1)        # (B,)

            # target net evaluates it
            q2_target = target(s2_t)                   # (B, A)
            q2_double = q2_target.gather(1, a2_online.view(-1,1)).squeeze(1)

            y = r_t + gamma * (1.0 - d_t) * q2_double


        loss = torch.nn.functional.smooth_l1_loss(q_sa, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        soft_update(target, model)

        loss_val = loss.item()
        loss_avg = (0.99 * loss_avg + 0.01 * loss_val) if loss_avg else loss_val

    # ---- main loop ----
    for ep in range(args.episodes):
        obs = env.reset(seed=(args.seed or 0) + ep)
        done, ret = False, 0.0

        while not done:
            # handle quit (single event pump for whole window)
            # for e in pg.event.get():
            #     if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
            #         done = True

            # ε-greedy action selection (override any fixed eps inside policy)
            eps = epsilon(global_step)
            if random.random() < eps:
                a = random.randrange(n_actions)
                # keep viewer “alive” even on random actions
                if policy.vis_callback is not None:
                    policy.vis_callback(activations=policy._acts,
                                        logits=np.zeros(n_actions, dtype=np.float32))
            else:
                a = policy.act(obs)  # uses model.eval() + no_grad internally

            # env step
            step = env.step(a)
            done_flag = float(step.terminated or step.truncated)
            replay.push(obs, a, step.reward, step.obs, done_flag)
            obs, ret = step.obs, ret + step.reward
            done = bool(done_flag)

            # draw both panes (single flip for entire window)
            # renderer.draw(step.info["snapshot"], score_text=True)
            # net_viewer.draw(fps=30)
            # pg.display.flip()
            # clock.tick(getattr(args, "fps", 12))

            # optimize & target update
            if len(replay) >= WARMUP and (global_step % OPT_EVERY == 0):
                optimize()

                if global_step < 50000:
                    optimize()
            # if global_step % TARGET_EVERY == 0:
            #     target.load_state_dict(model.state_dict())
            #     target.eval()

            global_step += 1

        print(f"[ep {ep:04d}] return={ret:.2f} score={step.info.get('score')} "
              f"reason={step.info.get('reason')} eps={epsilon(global_step):.3f} "
              f"loss~={loss_avg:.4f}")

    pg.quit()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["snake", "policy", "loop", "multi", "nnview"])
    p.add_argument("--grid-w", type=int, default=16)
    p.add_argument("--grid-h", type=int, default=16)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--cell-px", type=int, default=64)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--episodes-per-env", type=int, default=3)
    p.add_argument("--render", action="store_true")
    p.add_argument("--flat-dim", type=int, default=None, help="override input dim for NN viewer")
    p.add_argument("--n-actions", type=int, default=3, help="3=relative, 4=absolute")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "snake":
        run_snake(args)
    elif args.mode == "policy":
        run_policy(args)
    elif args.mode == "loop":
        run_loop(args)
    elif args.mode == "multi":
        run_multi(args)
    elif args.mode == "nnview":
        # default flat_dim if not provided
        if args.flat_dim is None:
            args.flat_dim = args.grid_w * args.grid_h * 3
        run_nn_viewer(args)



if __name__ == "__main__":
    main()
