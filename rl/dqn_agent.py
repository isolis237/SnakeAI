from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import os, json
import numpy as np

from core.interfaces import Policy
from .dqn_config import DQNConfig
from .networks import QNetwork
from .replay import ReplayBuffer, Transition
from .schedulers import EpsilonScheduler
from .utils import resolve_device

@dataclass
class DQNState:
    global_step: int = 0
    updates: int = 0


class DQNAgent(Policy):
    """Framework-agnostic DQN agent with a Torch-ready implementation path."""
    def __init__(
        self,
        q_net: QNetwork,
        target_net: QNetwork,
        replay: ReplayBuffer,
        eps_sched: EpsilonScheduler,
        cfg: DQNConfig(device=resolve_device(cfg.device)),
        optimizer: Any,  # e.g., torch.optim.Optimizer
        rng: Optional[np.random.Generator] = None,
    ):
        self.q_net = q_net
        self.target_net = target_net
        self.replay = replay
        self.eps_sched = eps_sched
        self.cfg = cfg
        self.optimizer = optimizer
        self.state = DQNState()
        self.rng = rng or np.random.default_rng(cfg.seed)

    # ------------- Policy protocol -------------
    def act(self, obs: np.ndarray) -> int:
        eps = self.eps_sched.value(self.state.global_step)
        if self.rng.random() < eps:
            return int(self.rng.integers(self.q_net.num_actions()))
        q = self.q_net.q_values(obs[None, ...])
        return int(np.argmax(q[0]))

    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        B = obs_batch.shape[0]
        A = self.q_net.num_actions()
        eps = self.eps_sched.value(self.state.global_step)
        explore = self.rng.random(B) < eps
        actions = np.empty((B,), dtype=np.int64)
        if explore.any():
            actions[explore] = self.rng.integers(A, size=int(explore.sum()))
        if (~explore).any():
            q = self.q_net.q_values(obs_batch[~explore])
            actions[~explore] = np.argmax(q, axis=1)
        return actions

    # ------------- Experience & learning -------------
    def observe(self, t: Transition) -> None:
        self.replay.add(t)
        self.state.global_step += 1

    def _stack_batch(self, batch: tuple[Transition, ...]) -> dict:
        obs = np.stack([t.obs for t in batch], axis=0).astype(np.float32)
        next_obs = np.stack([t.next_obs for t in batch], axis=0).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([bool(t.terminated or t.truncated) for t in batch], dtype=np.float32)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def update(self) -> Dict[str, Any] | None:
        if len(self.replay) < self.cfg.min_replay_before_learn:
            return None
        if (self.state.global_step % self.cfg.learn_every) != 0:
            return None

        # --- sample & stack ---
        batch = self.replay.sample(self.cfg.batch_size)
        B = len(batch)
        pack = self._stack_batch(batch)

        # --- torch bridge (import lazily to keep agent generic) ---
        import torch
        import torch.nn.functional as F

        device = torch.device(self.cfg.device)
        obs_t = torch.from_numpy(pack["obs"]).to(device)
        next_obs_t = torch.from_numpy(pack["next_obs"]).to(device)
        actions_t = torch.from_numpy(pack["actions"]).to(device)
        rewards_t = torch.from_numpy(pack["rewards"]).to(device)
        dones_t = torch.from_numpy(pack["dones"]).to(device)

        # --- Q(s,a) for taken actions ---
        q_online_all = self.q_net(obs_t)        
        q_sa = q_online_all.gather(1, actions_t.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                q_next_online = self.q_net(next_obs_t)          # online net to choose a*
                a_star = torch.argmax(q_next_online, dim=1)
                q_next_target = self.target_net(next_obs_t)     # target net to evaluate
                q_next = q_next_target.gather(1, a_star.view(-1, 1)).squeeze(1)
            else:
                q_next_target_all = self.target_net(next_obs_t)
                q_next, _ = q_next_target_all.max(dim=1)

            y = rewards_t + (1.0 - dones_t) * self.cfg.gamma * q_next

        # --- loss & optimize ---
        loss = F.smooth_l1_loss(q_sa, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        self.state.updates += 1

        # --- target updates ---
        if self.cfg.target_update == "hard":
            if (self.state.updates % self.cfg.target_update_interval) == 0:
                self.sync_target_hard()
        else:  # soft
            self.sync_target_soft(self.cfg.target_soft_tau)

        # --- stats ---
        stats = {
            "step": self.state.global_step,
            "updates": self.state.updates,
            "epsilon": float(self.eps_sched.value(self.state.global_step)),
            "loss": float(loss.detach().cpu().item()),
            "replay_size": int(len(self.replay)),
            "q_mean": float(q_online_all.mean().detach().cpu().item()),
            "q_max": float(q_online_all.max().detach().cpu().item()),
            "q_min": float(q_online_all.min().detach().cpu().item()),
        }
        return stats

    # ------------- Target sync -------------
    def sync_target_hard(self) -> None:
        self.target_net.hard_update_from(self.q_net)

    def sync_target_soft(self, tau: float | None = None) -> None:
        self.target_net.soft_update_from(self.q_net, tau or self.cfg.target_soft_tau)

    # ------------- Modes -------------
    def train_mode(self) -> None:
        self.q_net.train(); self.target_net.train()

    def eval_mode(self) -> None:
        self.q_net.eval(); self.target_net.eval()

    # ------------- Checkpointing -------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "cfg": asdict(self.cfg),
            "state": {"global_step": self.state.global_step, "updates": self.state.updates},
            "rng": self.rng.bit_generator.state,
            "replay": self.replay.get_state(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.state.global_step = int(state["state"]["global_step"])  # type: ignore[index]
        self.state.updates = int(state["state"]["updates"])          # type: ignore[index]
        self.rng.bit_generator.state = state["rng"]
        self.replay.set_state(state["replay"])  # lightweight restore

    def save(self, dirpath: str, tag: str = "dqn") -> None:
        os.makedirs(dirpath, exist_ok=True)
        # JSON runtime
        with open(os.path.join(dirpath, f"{tag}_agent_state.json"), "w") as f:
            json.dump(self.get_state(), f)
        # Delegate weights
        self.q_net.save_weights(os.path.join(dirpath, f"{tag}_online.pt"))
        self.target_net.save_weights(os.path.join(dirpath, f"{tag}_target.pt"))
        # Optimizer (if torch optimizer)
        try:
            import torch  # local import
            if hasattr(self.optimizer, "state_dict"):
                torch.save(self.optimizer.state_dict(), os.path.join(dirpath, f"{tag}_optim.pt"))
        except Exception:
            pass

    def load(self, dirpath: str, tag: str = "dqn") -> None:
        # JSON runtime
        with open(os.path.join(dirpath, f"{tag}_agent_state.json"), "r") as f:
            state = json.load(f)
        self.set_state(state)
        # Weights
        self.q_net.load_weights(os.path.join(dirpath, f"{tag}_online.pt"))
        self.target_net.load_weights(os.path.join(dirpath, f"{tag}_target.pt"))
        # Optimizer
        try:
            import torch
            if hasattr(self.optimizer, "load_state_dict"):
                self.optimizer.load_state_dict(torch.load(os.path.join(dirpath, f"{tag}_optim.pt"), map_location=self.cfg.device))
        except Exception:
            pass