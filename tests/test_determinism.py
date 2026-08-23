"""Same seed => bit-identical training trajectory. Non-determinism in an RL
codebase is a bug you find three weeks later, so it gets a test."""

import json

from ppo import Config, train


def _short_run(tmp_path, seed, name):
    cfg = Config(
        env_id="CartPole-v1",
        total_steps=2048,
        num_envs=4,
        rollout_len=64,
        num_minibatches=4,
        seed=seed,
        out_dir=str(tmp_path / name),
    )
    out = train(cfg)
    return [
        json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
    ]


def test_same_seed_identical_metrics(tmp_path):
    a = _short_run(tmp_path, seed=7, name="a")
    b = _short_run(tmp_path, seed=7, name="b")
    for ra, rb in zip(a, b):
        assert ra["pg_loss"] == rb["pg_loss"]
        assert ra["v_loss"] == rb["v_loss"]
        assert ra["entropy"] == rb["entropy"]


def test_different_seed_differs(tmp_path):
    a = _short_run(tmp_path, seed=7, name="c")
    b = _short_run(tmp_path, seed=8, name="d")
    assert any(ra["pg_loss"] != rb["pg_loss"] for ra, rb in zip(a, b))
