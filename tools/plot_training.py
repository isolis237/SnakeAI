# tools/plot_training.py
import math
import csv
from collections import deque
from pathlib import Path

# Use a non-interactive backend that writes to files
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Paths (resolve relative to repo root = parent of this script dir) ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOG_PATH = (REPO_ROOT / "runs" / "snake_dqn" / "logs.csv")  # <- canonical location
OUT_DIR = LOG_PATH.parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan

def rolling_mean(xs, window):
    out, q, s = [], deque(), 0.0
    for x in xs:
        x = float(x)
        q.append(x); s += x
        if len(q) > window:
            s -= q.popleft()
        out.append(s / len(q))
    return out

def rolling_rate(bools, window=100):
    out, q, s = [], deque(), 0.0
    for b in bools:
        v = 0.0 if math.isnan(b) else float(b)
        q.append(v); s += v
        if len(q) > window:
            s -= q.popleft()
        out.append(s / len(q))
    return out

if not LOG_PATH.exists():
    raise FileNotFoundError(f"Could not find logs.csv at {LOG_PATH}. "
                            f"Make sure you ran training and that your run writes to this path.")

steps = []
epis_reward = []; epis_reward_ema = []; epis_reward_mean100 = []
epis_len = []; epis_len_ema = []; epis_len_mean100 = []
final_score = []
death_wall = []; death_self = []; death_starv = []
train_loss, train_eps = [], []
train_qmean, train_qmax, train_qmin = [], [], []
ep_idx, ep_reward, ep_len = [], [], []

with LOG_PATH.open(newline="") as f:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames or []
    def get(row, k): return to_float(row[k]) if k in fieldnames and row.get(k) not in (None, "") else math.nan

    rows = 0
    for row in r:
        rows += 1
        steps.append(int(row["step"]))
        if row.get("episode") not in (None, ""):
            ep_idx.append(int(float(row["episode"])))
            ep_reward.append(to_float(row.get("epis/reward")))
            ep_len.append(to_float(row.get("epis/len")))
            
        epis_reward.append(get(row, "epis/reward"))
        epis_reward_ema.append(get(row, "epis/reward_ema"))
        epis_reward_mean100.append(get(row, "epis/reward_mean100"))
        epis_len.append(get(row, "epis/len"))
        epis_len_ema.append(get(row, "epis/len_ema"))
        epis_len_mean100.append(get(row, "epis/len_mean100"))
        final_score.append(get(row, "epis/final_score"))
        death_wall.append(get(row, "epis/death_wall"))
        death_self.append(get(row, "epis/death_self"))
        death_starv.append(get(row, "epis/death_starvation"))

        train_loss.append(get(row, "train/loss"))
        train_eps.append(get(row, "train/epsilon"))
        train_qmean.append(get(row, "train/q_mean"))
        train_qmax.append(get(row, "train/q_max"))
        train_qmin.append(get(row, "train/q_min"))

if rows == 0:
    raise RuntimeError(f"{LOG_PATH} has a header but no rows. Run training first, "
                       f"or lower warmup so it logs sooner.")

# Backfill EMA/mean@100 if missing
if all(math.isnan(x) for x in epis_reward_ema):
    epis_reward_ema = rolling_mean([0.0 if math.isnan(x) else x for x in epis_reward], window=20)
if all(math.isnan(x) for x in epis_len_ema):
    epis_len_ema = rolling_mean([0.0 if math.isnan(x) else x for x in epis_len], window=20)
if all(math.isnan(x) for x in epis_reward_mean100):
    epis_reward_mean100 = rolling_mean([0.0 if math.isnan(x) else x for x in epis_reward], window=100)
if all(math.isnan(x) for x in epis_len_mean100):
    epis_len_mean100 = rolling_mean([0.0 if math.isnan(x) else x for x in epis_len], window=100)

death_wall_r100 = rolling_rate(death_wall, 100)
death_self_r100 = rolling_rate(death_self, 100)
death_starv_r100 = rolling_rate(death_starv, 100)

# --- Individual plots ---
def savefig_named(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved: {path}")

fig = plt.figure(figsize=(10, 6))
plt.plot(steps, epis_reward, linewidth=1, alpha=0.5, label="raw")
plt.plot(steps, epis_reward_ema, linewidth=2, label="EMA(20)")
plt.plot(steps, epis_reward_mean100, linewidth=2, label="mean@100")
plt.title("Episode Reward"); plt.xlabel("step"); plt.ylabel("reward"); plt.legend()
savefig_named(fig, "epis_reward.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 6))
plt.plot(steps, epis_len, linewidth=1, alpha=0.5, label="raw")
plt.plot(steps, epis_len_ema, linewidth=2, label="EMA(20)")
plt.plot(steps, epis_len_mean100, linewidth=2, label="mean@100")
plt.title("Episode Length"); plt.xlabel("step"); plt.ylabel("steps/episode"); plt.legend()
savefig_named(fig, "epis_length.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 6))
plt.plot(steps, final_score, linewidth=2)
plt.title("Final Score per Episode"); plt.xlabel("step"); plt.ylabel("score")
savefig_named(fig, "epis_final_score.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 6))
plt.plot(steps, death_wall_r100, linewidth=2, label="wall")
plt.plot(steps, death_self_r100, linewidth=2, label="self")
plt.plot(steps, death_starv_r100, linewidth=2, label="starvation")
plt.title("Death Reasons (rolling 100)"); plt.xlabel("step"); plt.ylabel("fraction"); plt.legend()
savefig_named(fig, "epis_death_reasons.png"); plt.close(fig)

# Train metrics may be NaN for early runs; still plot what's available
fig = plt.figure(figsize=(10, 5))
plt.plot(steps, train_loss, linewidth=1)
plt.title("Train Loss"); plt.xlabel("step"); plt.ylabel("loss")
savefig_named(fig, "train_loss.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 5))
plt.plot(steps, train_eps, linewidth=1)
plt.title("Epsilon"); plt.xlabel("step"); plt.ylabel("epsilon")
savefig_named(fig, "train_epsilon.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 5))
plt.plot(ep_idx, ep_reward, linewidth=1, alpha=0.6)
plt.title("Episode Reward (by episode)"); plt.xlabel("episode"); plt.ylabel("reward")
savefig_named(fig, "epis_reward_by_episode.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 5))
plt.plot(ep_idx, ep_len, linewidth=1, alpha=0.6)
plt.title("Episode Length (by episode)"); plt.xlabel("episode"); plt.ylabel("steps/episode")
savefig_named(fig, "epis_length_by_episode.png"); plt.close(fig)

fig = plt.figure(figsize=(10, 5))
plt.plot(steps, train_qmean, label="mean")
plt.plot(steps, train_qmax, label="max")
plt.plot(steps, train_qmin, label="min")
plt.title("Q stats"); plt.xlabel("step"); plt.ylabel("Q"); plt.legend()
savefig_named(fig, "train_qstats.png"); plt.close(fig)

# --- Combined dashboard (optional) ---
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

ax = fig.add_subplot(gs[0, 0])
ax.plot(steps, epis_reward, alpha=0.4); ax.plot(steps, epis_reward_ema); ax.plot(steps, epis_reward_mean100)
ax.set_title("Episode Reward"); ax.set_xlabel("step"); ax.set_ylabel("reward")

ax = fig.add_subplot(gs[0, 1])
ax.plot(steps, epis_len, alpha=0.4); ax.plot(steps, epis_len_ema); ax.plot(steps, epis_len_mean100)
ax.set_title("Episode Length"); ax.set_xlabel("step"); ax.set_ylabel("steps")

ax = fig.add_subplot(gs[1, 0])
ax.plot(steps, final_score); ax.set_title("Final Score"); ax.set_xlabel("step"); ax.set_ylabel("score")

ax = fig.add_subplot(gs[1, 1])
ax.plot(steps, death_wall_r100, label="wall"); ax.plot(steps, death_self_r100, label="self"); ax.plot(steps, death_starv_r100, label="starvation"); ax.legend()
ax.set_title("Death Reasons (rolling 100)"); ax.set_xlabel("step"); ax.set_ylabel("fraction")

ax = fig.add_subplot(gs[2, 0])
ax.plot(steps, train_loss); ax.set_title("Train Loss"); ax.set_xlabel("step"); ax.set_ylabel("loss")

ax = fig.add_subplot(gs[2, 1])
ax.plot(steps, train_eps); ax.set_title("Epsilon"); ax.set_xlabel("step"); ax.set_ylabel("epsilon")

savefig_named(fig, "training_summary.png"); plt.close(fig)

print(f"\nAll plots saved under: {OUT_DIR.resolve()}")
