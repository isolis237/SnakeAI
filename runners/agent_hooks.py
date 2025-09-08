from __future__ import annotations
from typing import TYPE_CHECKING
from rl.logging import MultiLogger
from rl.metrics import EMA, WindowedStat

if TYPE_CHECKING:
    from config import AppConfig
    from rl.dqn_agent import DQNAgent


ALL_KEYS = [
    "step",
    "train/loss","train/epsilon","train/replay_size","train/q_mean","train/q_max","train/q_min",
    "train/updates","train/grad_norm",
    "train/learn_started","train/replay_fill",
    "epis/reward","epis/reward_ema","epis/reward_mean100",
    "epis/len","epis/len_ema","epis/len_mean100",
    "epis/final_score","epis/death_wall","epis/death_self","epis/death_starvation",
]

class AgentHooks:
    def __init__(self, cfg: "AppConfig", logger: MultiLogger, agent: "DQNAgent"):
        self.cfg = cfg
        self.logger = logger
        self.agent = agent
        self.ema_reward = EMA(0.05)
        self.ema_length = EMA(0.05)
        self.win_reward = WindowedStat(100)
        self.win_length = WindowedStat(100)

    def on_step_log(self, stats):
        if stats.get("updates", 0) % 10 != 0 and stats.get("step", 0) % 200 != 0:
            return

        scalars = {
            "train/loss": stats.get("loss"),
            "train/epsilon": stats.get("epsilon"),
            "train/replay_size": stats.get("replay_size"),
            "train/q_mean": stats.get("q_mean"),
            "train/q_max": stats.get("q_max"),
            "train/q_min": stats.get("q_min"),
            "train/updates": stats.get("updates"),
            "train/grad_norm": stats.get("grad_norm"),
            "train/learn_started": 1.0 if stats.get("replay_size", 0) >= self.cfg.dqn.min_replay_before_learn else 0.0,
            "train/replay_fill": stats.get("replay_size"),
        }
        self.logger.log(stats["step"], scalars)

    def on_episode_end(self, ep, s):
        step = self.agent.state.global_step
        er, el = s["reward"], s["steps"]
        r_ema = self.ema_reward.update(er); l_ema = self.ema_length.update(el)
        self.win_reward.add(er); self.win_length.add(el)
        wr, wl = self.win_reward.summary(), self.win_length.summary()
        self.logger.log(step, {
            "episode": ep,
            "epis/reward": er,
            "epis/reward_ema": r_ema,
            "epis/reward_mean100": wr["mean"],
            "epis/len": el,
            "epis/len_ema": l_ema,
            "epis/len_mean100": wl["mean"],
            "epis/final_score": s.get("final_score", 0),
            "epis/death_wall": 1.0 if s.get("death_reason") == "wall" else 0.0,
            "epis/death_self": 1.0 if s.get("death_reason") == "self" else 0.0,
            "epis/death_starvation": 1.0 if s.get("death_reason") == "starvation" else 0.0,
        })
        self.logger.flush()
