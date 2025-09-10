from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from core.interfaces import Env, Transition
from .dqn_agent import DQNAgent
from rl.logging import CSVLogger
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
        live_hook: Optional[Any] = None
    ):
        self.env = env
        self.agent = agent
        self.hooks = hooks or TrainHooks()
        self.max_steps_per_episode = max_steps_per_episode
        self.render_every = render_every
        self.live_hook = live_hook

    def train_and_stream(self, num_episodes: int) -> None:
        """
        Train while producing full-episode streams/recordings.
        This does NOT block on rendering: the trainer only *collects* snapshots
        and hands them to the live_hook (which may be an async episode player).
        """
        if self.live_hook is not None:
            self.live_hook.ensure_started()

        try:
            for ep in range(num_episodes):
                # Announce the episode so the hook decides whether to record/stream it.
                if self.live_hook is not None:
                    self.live_hook.on_episode_start(ep)

                obs = self.env.reset()
                done = False
                steps = 0
                ep_reward = 0.0
                last_step = None

                # Record the initial state if this episode is selected.
                if self.live_hook is not None and self.live_hook.is_streaming():
                    try:
                        snap0 = self.env.get_snapshot()
                        # Prefer new API if available
                        if hasattr(self.live_hook, "collect_snapshot"):
                            self.live_hook.collect_snapshot(snap0)
                        else:
                            self.live_hook.maybe_render(None, {"snapshot": snap0})
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

                    # Record a snapshot for THIS step if we’re streaming/recording this episode.
                    if self.live_hook is not None and self.live_hook.is_streaming():
                        try:
                            snap = self.env.get_snapshot()
                            if hasattr(self.live_hook, "collect_snapshot"):
                                self.live_hook.collect_snapshot(snap)
                            else:
                                # Back-compat with your original hook
                                self.live_hook.maybe_render(None, {"snapshot": snap})
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
                final_reason = final_info.get("reason")
                final_score  = final_info.get("score", 0)

                # Let the hook enqueue or finalize the episode
                if self.live_hook is not None:
                    self.live_hook.on_episode_end({
                        "steps": steps,
                        "reward": ep_reward,
                        "final_score": final_score,
                        "death_reason": final_reason,
                    }, final_info)

                # Your logging callbacks
                if self.hooks.on_episode_end:
                    self.hooks.on_episode_end(ep, {
                        "steps": steps,
                        "reward": ep_reward,
                        "final_score": final_score,
                        "death_reason": final_reason,
                    })

            # Optional: replay best after the last episode is done
            if self.live_hook is not None:
                self.live_hook.replay_best()

        finally:
            # Ensure viewer/resources are closed even if training errors out.
            if self.live_hook is not None:
                try:
                    # If using the async player LiveRenderHook, this will drain & close cleanly.
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
