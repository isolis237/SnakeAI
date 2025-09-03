from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from core.interfaces import Env
from .dqn_agent import DQNAgent
from .replay import Transition
from rl.logging import CSVLogger, TBLogger, MultiLogger
from rl.metrics import EMA, WindowedStat

LogFn = Callable[[Dict[str, Any]], None]

@dataclass
class TrainHooks:
    on_episode_end: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_step_log: Optional[LogFn] = None
    on_eval_end: Optional[Callable[[int, Dict[str, Any]], None]] = None

class DQNTrainer:
    """
    Thin, testable loop coordinator. Keeps training loop separate from agent logic.
    """
    def __init__(
        self,
        env: Env,
        agent: DQNAgent,
        hooks: Optional[TrainHooks] = None,
        max_steps_per_episode: Optional[int] = None,
        render_every: Optional[int] = None,  # if you want to render periodically
    ):
        self.env = env
        self.agent = agent
        self.hooks = hooks or TrainHooks()
        self.max_steps_per_episode = max_steps_per_episode
        self.render_every = render_every

    def train(self, num_episodes: int) -> None:
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            steps = 0
            ep_reward = 0.0
            step = None

            while not done:
                action = self.agent.act(obs)
                step = self.env.step(action)
                self.agent.observe(Transition(
                    obs=obs, action=action, reward=step.reward,
                    next_obs=step.obs, terminated=step.terminated,
                    truncated=step.truncated, info=step.info
                ))
                stats = self.agent.update()
                if stats and self.hooks.on_step_log:
                    self.hooks.on_step_log(stats)

                obs = step.obs
                done = step.terminated or step.truncated
                steps += 1
                ep_reward += float(step.reward)

                if self.max_steps_per_episode and steps >= self.max_steps_per_episode:
                    break

            final_reason = step.info.get("reason") if "step" in locals() else None
            final_score  = step.info.get("score") if "step" in locals() else 0

            if self.hooks.on_episode_end:
                self.hooks.on_episode_end(ep, {
                    "steps": steps,
                    "reward": ep_reward,
                    "final_score": final_score,
                    "death_reason": final_reason,
                })

    def evaluate(self, num_episodes: int) -> Dict[str, Any]:
        """
        Greedy evaluation sketch; switch agent to eval mode before calling if needed.
        """
        # TODO: implement when needed
        return {}
