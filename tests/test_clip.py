"""The asymmetric clip objective: equivalence to PPO when symmetric, and the
exact geometry of its dead zones (where the gradient must vanish)."""

import torch

from ppo import ppo_policy_loss


def reference_symmetric_ppo(ratio, adv, eps):
    """Textbook PPO surrogate, written independently of the implementation."""
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()


def test_symmetric_case_is_exactly_ppo():
    torch.manual_seed(0)
    ratio = torch.exp(torch.randn(1000) * 0.3)
    adv = torch.randn(1000)
    ours = ppo_policy_loss(ratio, adv, clip_low=0.2, clip_high=0.2)
    ref = reference_symmetric_ppo(ratio, adv, eps=0.2)
    assert torch.allclose(ours, ref, atol=1e-7)


def _grad_at(logratio_value, adv, clip_low, clip_high):
    x = torch.tensor([logratio_value], requires_grad=True)
    loss = ppo_policy_loss(torch.exp(x), torch.tensor([adv]), clip_low, clip_high)
    loss.backward()
    return x.grad.item()


def test_dead_zone_positive_advantage_above_clip_high():
    # A > 0 and ratio above 1 + clip_high: PPO stops pushing the action up.
    assert _grad_at(0.5, adv=1.0, clip_low=0.2, clip_high=0.3) == 0.0
    # Just below the bound the gradient is alive.
    assert _grad_at(0.2, adv=1.0, clip_low=0.2, clip_high=0.3) != 0.0


def test_dead_zone_negative_advantage_below_clip_low():
    # A < 0 and ratio below 1 - clip_low: PPO stops pushing the action down.
    # THIS is the zone Park et al. tie to entropy: while active, it keeps
    # suppressing low-probability actions; clipping cuts it off.
    assert _grad_at(-0.5, adv=-1.0, clip_low=0.3, clip_high=0.2) == 0.0
    assert _grad_at(-0.2, adv=-1.0, clip_low=0.3, clip_high=0.2) != 0.0


def test_pessimism_no_dead_zone_on_the_bad_side():
    # A > 0, ratio far BELOW 1 - clip_low: the min() keeps the unclipped term,
    # gradient stays alive (clipping is one-sided per advantage sign).
    assert _grad_at(-0.5, adv=1.0, clip_low=0.2, clip_high=0.2) != 0.0
    # A < 0, ratio far ABOVE 1 + clip_high: same, still alive.
    assert _grad_at(0.5, adv=-1.0, clip_low=0.2, clip_high=0.2) != 0.0


def test_asymmetry_moves_only_its_own_bound():
    torch.manual_seed(1)
    ratio = torch.exp(torch.randn(4096) * 0.4)
    adv = torch.randn(4096)
    base = ppo_policy_loss(ratio, adv, 0.2, 0.2)
    widened_high = ppo_policy_loss(ratio, adv, 0.2, 0.5)
    widened_low = ppo_policy_loss(ratio, adv, 0.5, 0.2)
    # Widening a bound can only change samples that were clipped at that bound.
    assert not torch.allclose(base, widened_high)
    assert not torch.allclose(base, widened_low)
    # And with ratios forced inside [0.8, 1.2], the bounds are irrelevant.
    inside = torch.clamp(ratio, 0.85, 1.15)
    assert torch.allclose(
        ppo_policy_loss(inside, adv, 0.2, 0.2), ppo_policy_loss(inside, adv, 0.9, 0.9)
    )
