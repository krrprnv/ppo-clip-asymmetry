"""Vectorized environments with one tiny API for two backends.

Backends:
  * Gymnasium classic control ("CartPole-v1", "Acrobot-v1", ...) — vector obs.
  * MinAtar ("minatar/breakout", "minatar/asterix", ...) — 10x10xC binary
    images, returned channel-first as float32 so they feed a conv net directly.

Both are wrapped in SyncVecEnv, which steps N copies in a python loop and
autoresets on episode end (next-step autoreset: the obs returned after a
terminal step is the FIRST obs of the next episode). The true last
observation of a finished episode is returned separately in `final_obs`,
because PPO must bootstrap V(s_final) when an episode was truncated rather
than terminated. Writing this ~150-line wrapper by hand beats fighting the
autoreset semantics that changed between gymnasium versions.
"""

from __future__ import annotations

import numpy as np

MINATAR_PREFIX = "minatar/"
# MinAtar episodes have no built-in time limit; freeway can in principle go
# forever with a bad policy. A generous cap keeps rollouts finite.
MINATAR_MAX_STEPS = 10_000


class MinAtarGym:
    """Adapts minatar.Environment to the gymnasium (obs, r, term, trunc) step API."""

    def __init__(self, game: str, sticky_action_prob: float = 0.1):
        from minatar import Environment

        self._env = Environment(game, sticky_action_prob=sticky_action_prob)
        c = self._env.state_shape()[2]
        self.obs_shape = (c, 10, 10)
        self.n_actions = self._env.num_actions()
        self._steps = 0

    def _obs(self) -> np.ndarray:
        # (10, 10, C) bool -> (C, 10, 10) float32
        return self._env.state().transpose(2, 0, 1).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._env.seed(seed)
        self._env.reset()
        self._steps = 0
        return self._obs()

    def step(self, action: int):
        reward, terminated = self._env.act(int(action))
        self._steps += 1
        truncated = (not terminated) and self._steps >= MINATAR_MAX_STEPS
        return self._obs(), float(reward), bool(terminated), truncated


class GymAdapter:
    """Thin adapter so single gymnasium envs match MinAtarGym's surface."""

    def __init__(self, env_id: str):
        import gymnasium as gym

        self._env = gym.make(env_id)
        self.obs_shape = self._env.observation_space.shape
        self.n_actions = self._env.action_space.n

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self._env.reset(seed=seed)
        return obs.astype(np.float32)

    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self._env.step(int(action))
        return obs.astype(np.float32), float(reward), terminated, truncated


def _make_one(env_id: str, sticky_action_prob: float):
    if env_id.startswith(MINATAR_PREFIX):
        return MinAtarGym(env_id[len(MINATAR_PREFIX) :], sticky_action_prob)
    return GymAdapter(env_id)


class SyncVecEnv:
    """N synchronous envs, next-step autoreset, truncation-aware.

    step(actions) returns:
      obs        (N, *obs_shape) float32 — post-autoreset observations
      rewards    (N,) float32
      terminated (N,) bool — env reached a true terminal state (V(s')=0)
      truncated  (N,) bool — episode cut off (V(s') must be bootstrapped)
      final_obs  (N, *obs_shape) float32 — valid only where terminated|truncated:
                 the actual last observation, before the autoreset
      finished   list of (return, length) for episodes that ended this step
    """

    def __init__(
        self, env_id: str, num_envs: int, seed: int, sticky_action_prob: float = 0.1
    ):
        self.envs = [_make_one(env_id, sticky_action_prob) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.obs_shape = self.envs[0].obs_shape
        self.n_actions = self.envs[0].n_actions
        self._seeds = [seed + i for i in range(num_envs)]
        self._ep_ret = np.zeros(num_envs, dtype=np.float64)
        self._ep_len = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> np.ndarray:
        obs = np.stack([e.reset(seed=s) for e, s in zip(self.envs, self._seeds)])
        self._ep_ret[:] = 0.0
        self._ep_len[:] = 0
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        n = self.num_envs
        obs = np.zeros((n, *self.obs_shape), dtype=np.float32)
        final_obs = np.zeros((n, *self.obs_shape), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        finished: list[tuple[float, int]] = []

        for i, env in enumerate(self.envs):
            (
                o,
                r,
                term,
                trunc,
            ) = env.step(actions[i])
            rewards[i] = r
            self._ep_ret[i] += r
            self._ep_len[i] += 1
            if term or trunc:
                terminated[i] = term
                truncated[i] = trunc
                final_obs[i] = o
                finished.append((float(self._ep_ret[i]), int(self._ep_len[i])))
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0
                obs[i] = env.reset()  # seeded once at start; RNG state carries over
            else:
                obs[i] = o

        return obs, rewards, terminated, truncated, final_obs, finished
