"""PPO from scratch, with independently tunable lower/upper clip bounds.

The only non-standard thing in this file is that the clip range is split:

    L(theta) = E[ min( r_t(theta) * A_t,
                       clip(r_t(theta), 1 - clip_low, 1 + clip_high) * A_t ) ]

clip_low = clip_high = 0.2 recovers textbook PPO exactly. Park et al.
(arXiv:2509.26114) report, in LLM RL, that loosening clip_low RAISES policy
entropy while loosening clip_high LOWERS it — DAPO's "clip-higher" trick
rides on the second half. Both were discovered on ~100k-token vocabularies.
This implementation exists to test whether the mechanism operates at |A|=6,
in the environments PPO came from, which is why ent_coef defaults to 0:
if entropy moves, it has to be the clip moving it.

Everything else is deliberately boring: GAE(lambda), advantage normalization
per minibatch, orthogonal init, LR annealing, grad-norm clipping — the
standard recipe, so that clip asymmetry is the only experimental variable.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from envs import SyncVecEnv


@dataclass
class Config:
    env_id: str = "CartPole-v1"  # gymnasium id, or "minatar/<game>"
    total_steps: int = 1_000_000
    num_envs: int = 32
    rollout_len: int = 128
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_low: float = 0.2  # ratio clipped below at 1 - clip_low
    clip_high: float = 0.2  # ratio clipped above at 1 + clip_high
    epochs: int = 4
    num_minibatches: int = 4
    ent_coef: float = 0.0  # 0 on purpose — see module docstring
    vf_coef: float = 0.5
    clip_vloss: bool = True
    vf_clip: float = 0.2  # value clip range, fixed: NOT tied to the policy clip
    max_grad_norm: float = 0.5
    sticky_action_prob: float = 0.1  # MinAtar only
    seed: int = 1
    torch_threads: int = 0  # 0 = leave torch's default
    device: str = "cpu"
    out_dir: str = "runs/dev"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_len

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def _ortho(layer: nn.Linear | nn.Conv2d, std: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLPActorCritic(nn.Module):
    """Separate 64-64 tanh trunks for vector observations (classic control)."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            _ortho(nn.Linear(64, 64)),
            nn.Tanh(),
            _ortho(nn.Linear(64, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            _ortho(nn.Linear(64, 64)),
            nn.Tanh(),
            _ortho(nn.Linear(64, 1), std=1.0),
        )

    def logits_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)


class ConvActorCritic(nn.Module):
    """Shared conv trunk + two heads for MinAtar's (C, 10, 10) binary frames."""

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            _ortho(nn.Conv2d(in_channels, 16, kernel_size=3)),
            nn.ReLU(),
            nn.Flatten(),
            _ortho(nn.Linear(16 * 8 * 8, 128)),
            nn.ReLU(),
        )
        self.policy_head = _ortho(nn.Linear(128, n_actions), std=0.01)
        self.value_head = _ortho(nn.Linear(128, 1), std=1.0)

    def logits_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def make_agent(obs_shape: tuple[int, ...], n_actions: int) -> nn.Module:
    if len(obs_shape) == 3:
        return ConvActorCritic(obs_shape[0], n_actions)
    return MLPActorCritic(int(np.prod(obs_shape)), n_actions)


def compute_gae(
    rewards: torch.Tensor,  # (T, N)
    values: torch.Tensor,  # (T, N)
    dones: torch.Tensor,  # (T, N) — episode ended AT step t (term or trunc)
    next_value: torch.Tensor,  # (N,) — V(s_{T}) for bootstrapping the last step
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Pure function so the test can check
    it against hand-computed values.

    `dones[t] = 1` means the transition at t ended its episode, so nothing
    beyond t is bootstrapped through it. Truncated episodes must have had
    gamma * V(s_final) folded into rewards[t] BEFORE calling this (the
    training loop does that), which is what makes treating truncation as
    "done" correct here.
    """
    T, _ = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalue = next_value
        else:
            nextvalue = values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalue * nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def ppo_policy_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float,
    clip_high: float,
) -> torch.Tensor:
    """Asymmetric clipped surrogate (pessimistic max over the two candidates)."""
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    return torch.max(pg1, pg2).mean()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: Config) -> Path:
    assert cfg.batch_size % cfg.num_minibatches == 0
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    envs = SyncVecEnv(
        cfg.env_id,
        cfg.num_envs,
        seed=cfg.seed,
        sticky_action_prob=cfg.sticky_action_prob,
    )
    agent = make_agent(envs.obs_shape, envs.n_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.rollout_len, cfg.num_envs
    obs_buf = torch.zeros((T, N, *envs.obs_shape), device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    obs = torch.as_tensor(envs.reset(), device=device)
    num_updates = cfg.total_steps // cfg.batch_size
    recent_returns: list[float] = []
    global_step = 0
    start = time.time()
    metrics_path = out / "metrics.jsonl"
    metrics_file = metrics_path.open("w")

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.lr

        # ---- collect one rollout ------------------------------------------
        rollout_entropies = []
        for t in range(T):
            global_step += N
            with torch.no_grad():
                logits, value = agent.logits_and_value(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                rollout_entropies.append(dist.entropy().mean().item())
            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = dist.log_prob(action)
            val_buf[t] = value

            next_obs, reward, terminated, truncated, final_obs, finished = envs.step(
                action.cpu().numpy()
            )
            reward = torch.as_tensor(reward, device=device)

            # Truncation is not death: fold gamma * V(s_final) into the reward
            # so GAE can treat every episode boundary uniformly as "done".
            trunc_only = truncated & ~terminated
            if trunc_only.any():
                idx = np.flatnonzero(trunc_only)
                with torch.no_grad():
                    _, v_final = agent.logits_and_value(
                        torch.as_tensor(final_obs[idx], device=device)
                    )
                reward[idx] += cfg.gamma * v_final

            rew_buf[t] = reward
            done_buf[t] = torch.as_tensor(
                terminated | truncated, dtype=torch.float32, device=device
            )
            obs = torch.as_tensor(next_obs, device=device)
            recent_returns.extend(r for r, _ in finished)

        recent_returns = recent_returns[-100:]
        with torch.no_grad():
            _, next_value = agent.logits_and_value(obs)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, cfg.gamma, cfg.gae_lambda
        )

        # ---- update -------------------------------------------------------
        b_obs = obs_buf.reshape(-1, *envs.obs_shape)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        idx = np.arange(cfg.batch_size)
        kl_sum, clip_lo_sum, clip_hi_sum, n_mb = 0.0, 0.0, 0.0, 0
        pg_loss_v = v_loss_v = ent_v = 0.0
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for mb_start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[mb_start : mb_start + cfg.minibatch_size]
                logits, value = agent.logits_and_value(b_obs[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                entropy = dist.entropy().mean()
                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = ppo_policy_loss(ratio, mb_adv, cfg.clip_low, cfg.clip_high)

                if cfg.clip_vloss:
                    v_clipped = b_val[mb] + torch.clamp(
                        value - b_val[mb], -cfg.vf_clip, cfg.vf_clip
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            (value - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * ((value - b_ret[mb]) ** 2).mean()

                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    kl_sum += ((ratio - 1) - logratio).mean().item()  # k3 estimator
                    clip_lo_sum += (ratio < 1.0 - cfg.clip_low).float().mean().item()
                    clip_hi_sum += (ratio > 1.0 + cfg.clip_high).float().mean().item()
                n_mb += 1
                pg_loss_v, v_loss_v, ent_v = (
                    pg_loss.item(),
                    v_loss.item(),
                    entropy.item(),
                )

        record = {
            "update": update,
            "global_step": global_step,
            "episodic_return": float(np.mean(recent_returns))
            if recent_returns
            else None,
            "entropy": float(np.mean(rollout_entropies)),  # behavior-policy entropy
            "approx_kl": kl_sum / n_mb,
            "clipfrac_low": clip_lo_sum / n_mb,
            "clipfrac_high": clip_hi_sum / n_mb,
            "pg_loss": pg_loss_v,
            "v_loss": v_loss_v,
            "update_entropy": ent_v,
            "lr": optimizer.param_groups[0]["lr"],
            "sps": int(global_step / (time.time() - start)),
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    metrics_file.close()
    torch.save(
        {"agent": agent.state_dict(), "config": asdict(cfg)}, out / "checkpoint.pt"
    )
    return out
