# rl/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from core.interfaces import Env, Transition
from .dqn_agent import DQNAgent

LogFn = Callable[[Dict[str, Any]], None]

@dataclass
class TrainHooks:
    on_episode_end: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_step_log: Optional[LogFn] = None
    on_eval_end: Optional[Callable[[int, Dict[str, Any]], None]] = None

class DQNTrainer:
    def __init__(
        self,
        env: Env,
        agent: DQNAgent,
        hooks: Optional[TrainHooks] = None,
        max_steps_per_episode: Optional[int] = None,
        render_every: Optional[int] = None,
        live_hook: Optional[Any] = None,
    ):
        self.env = env
        self.agent = agent
        self.hooks = hooks or TrainHooks()
        self.max_steps_per_episode = max_steps_per_episode
        self.render_every = max(1, int(render_every)) if render_every else 1
        self.live_hook = live_hook

    def train_and_stream(self, num_episodes: int) -> None:
        # Start live hook once if present
        if self.live_hook is not None and hasattr(self.live_hook, "ensure_started"):
            self.live_hook.ensure_started()

        try:
            for ep in range(num_episodes):
                if self.live_hook is not None and hasattr(self.live_hook, "start_episode"):
                    self.live_hook.start_episode(ep)

                obs = self.env.reset()
                done = False
                steps = 0
                ep_reward = 0.0
                last_step = None

                # initial frame
                if self.live_hook is not None and hasattr(self.live_hook, "record_snapshot"):
                    try:
                        snap0 = self.env.get_snapshot()
                        self.live_hook.record_snapshot(snap0)
                    except Exception:
                        pass

                while not done:
                    action = self.agent.act(obs)
                    step = self.env.step(action)
                    last_step = step

                    # learning
                    self.agent.observe(Transition(
                        obs=obs, action=action, reward=step.reward,
                        next_obs=step.obs, terminated=step.terminated,
                        truncated=step.truncated, info=step.info
                    ))
                    stats = self.agent.update()
                    if stats and self.hooks.on_step_log:
                        self.hooks.on_step_log(stats)

                    # capture snapshot every N steps (let hook decide whether to store)
                    if self.live_hook is not None and (steps % self.render_every == 0):
                        if hasattr(self.live_hook, "record_snapshot"):
                            try:
                                snap = self.env.get_snapshot()
                                self.live_hook.record_snapshot(snap)
                            except Exception:
                                pass

                    # episode bookkeeping
                    obs = step.obs
                    done = step.terminated or step.truncated
                    steps += 1
                    ep_reward += float(step.reward)

                    if self.max_steps_per_episode and steps >= self.max_steps_per_episode:
                        break

                final_info = (last_step.info if last_step is not None else {}) or {}
                summary = {
                    "steps": steps,
                    "reward": ep_reward,
                    "final_score": final_info.get("score", 0),
                    "death_reason": final_info.get("reason"),
                }

                if self.live_hook is not None and hasattr(self.live_hook, "end_episode"):
                    self.live_hook.end_episode(summary, final_info)

                if self.hooks.on_episode_end:
                    self.hooks.on_episode_end(ep, summary)

            if self.live_hook is not None and hasattr(self.live_hook, "replay_best_and_wait"):
                self.live_hook.replay_best_and_wait(timeout=15.0)
            elif self.live_hook is not None and hasattr(self.live_hook, "replay_best"):
                # fallback: non-blocking, may still get cut if you close immediately
                self.live_hook.replay_best()

        finally:
            if self.live_hook is not None and hasattr(self.live_hook, "close"):
                try:
                    self.live_hook.close()
                except Exception:
                    pass


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
