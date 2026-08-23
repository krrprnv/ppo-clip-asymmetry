"""CLI entry point: one training run, one output directory.

Examples:
    uv run python -m ppo.train --env CartPole-v1 --total-steps 100000 --ent-coef 0.01
    uv run python -m ppo.train --env minatar/breakout --clip-low 0.05 --clip-high 0.5 \
        --total-steps 1000000 --seed 3 --out runs/breakout/cl0.05_ch0.5/seed3
"""

from __future__ import annotations

import argparse
import dataclasses

from ppo.core import Config, train


def parse_args() -> Config:
    defaults = Config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default=defaults.env_id, dest="env_id")
    p.add_argument("--total-steps", type=int, default=defaults.total_steps)
    p.add_argument("--num-envs", type=int, default=defaults.num_envs)
    p.add_argument("--rollout-len", type=int, default=defaults.rollout_len)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--no-anneal-lr", action="store_false", dest="anneal_lr")
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    p.add_argument("--clip-low", type=float, default=defaults.clip_low)
    p.add_argument("--clip-high", type=float, default=defaults.clip_high)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--num-minibatches", type=int, default=defaults.num_minibatches)
    p.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    p.add_argument("--vf-coef", type=float, default=defaults.vf_coef)
    p.add_argument("--no-clip-vloss", action="store_false", dest="clip_vloss")
    p.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    p.add_argument(
        "--sticky",
        type=float,
        default=defaults.sticky_action_prob,
        dest="sticky_action_prob",
    )
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--out", default=defaults.out_dir, dest="out_dir")
    args = p.parse_args()

    field_names = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in vars(args).items() if k in field_names})


if __name__ == "__main__":
    cfg = parse_args()
    out = train(cfg)
    print(f"done: {out}")
