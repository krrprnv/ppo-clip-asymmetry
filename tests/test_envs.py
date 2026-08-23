"""The vectorized wrapper: shapes, autoreset, episode accounting."""

import numpy as np

from ppo.envs import SyncVecEnv


def test_cartpole_shapes_and_dtypes():
    envs = SyncVecEnv("CartPole-v1", num_envs=3, seed=0)
    obs = envs.reset()
    assert obs.shape == (3, 4) and obs.dtype == np.float32
    obs, r, term, trunc, final_obs, finished = envs.step(np.zeros(3, dtype=np.int64))
    assert obs.shape == (3, 4) and r.shape == (3,)
    assert term.dtype == bool and trunc.dtype == bool
    assert final_obs.shape == (3, 4)


def test_minatar_obs_is_channel_first_binary():
    envs = SyncVecEnv("minatar/breakout", num_envs=2, seed=0)
    obs = envs.reset()
    c = envs.obs_shape[0]
    assert obs.shape == (2, c, 10, 10)
    assert envs.n_actions == 6
    assert set(np.unique(obs)).issubset({0.0, 1.0})


def test_autoreset_reports_episodes_and_restarts():
    envs = SyncVecEnv("CartPole-v1", num_envs=2, seed=0)
    envs.reset()
    total_episodes = 0
    rng = np.random.default_rng(0)
    for _ in range(600):  # CartPole under random actions dies fast
        actions = rng.integers(0, 2, size=2)
        _, _, term, trunc, _, finished = envs.step(actions)
        total_episodes += len(finished)
        for ret, length in finished:
            assert length >= 1 and ret >= 1.0  # CartPole pays +1 per step
    assert total_episodes > 0  # episodes ended and the wrapper kept going


def test_episode_length_matches_return_for_cartpole():
    # CartPole reward is exactly +1 per step, so return == length always.
    envs = SyncVecEnv("CartPole-v1", num_envs=1, seed=1)
    envs.reset()
    rng = np.random.default_rng(1)
    for _ in range(2000):
        _, _, _, _, _, finished = envs.step(rng.integers(0, 2, size=1))
        for ret, length in finished:
            assert ret == float(length)
