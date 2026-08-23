"""GAE checked against values computed by hand (calculator, not code)."""

import torch

from ppo import compute_gae

GAMMA = 0.5
LAM = 0.5


def test_gae_matches_hand_computation():
    # Single env, 3 steps, no episode boundary.
    # rewards = [1, 1, 1], values = [2, 2, 2], V(s_3) = 2.
    # delta_t = r + gamma*V' - V = 1 + 0.5*2 - 2 = 0 for every t
    # => all advantages 0, returns == values.
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[2.0], [2.0], [2.0]])
    dones = torch.zeros(3, 1)
    adv, ret = compute_gae(rewards, values, dones, torch.tensor([2.0]), GAMMA, LAM)
    assert torch.allclose(adv, torch.zeros(3, 1), atol=1e-6)
    assert torch.allclose(ret, values, atol=1e-6)


def test_gae_recursion_by_hand():
    # rewards = [0, 1], values = [1, 1], V(s_2) = 4, no dones.
    # delta_1 = 1 + 0.5*4 - 1 = 2
    # delta_0 = 0 + 0.5*1 - 1 = -0.5
    # A_1 = 2
    # A_0 = delta_0 + gamma*lam*A_1 = -0.5 + 0.25*2 = 0
    rewards = torch.tensor([[0.0], [1.0]])
    values = torch.tensor([[1.0], [1.0]])
    dones = torch.zeros(2, 1)
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([4.0]), GAMMA, LAM)
    assert torch.allclose(adv, torch.tensor([[0.0], [2.0]]), atol=1e-6)


def test_done_blocks_bootstrap_and_credit():
    # Episode ends at t=0: nothing after t=0 may leak into A_0.
    # rewards = [1, 100], values = [3, 0], dones = [1, 0], V(s_2) = 100.
    # delta_0 = 1 + 0 - 3 = -2 (done => no bootstrap), A_0 = -2 exactly.
    rewards = torch.tensor([[1.0], [100.0]])
    values = torch.tensor([[3.0], [0.0]])
    dones = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([100.0]), GAMMA, LAM)
    assert torch.allclose(adv[0], torch.tensor([-2.0]), atol=1e-6)


def test_shapes_multi_env():
    T, N = 7, 3
    adv, ret = compute_gae(
        torch.rand(T, N),
        torch.rand(T, N),
        torch.zeros(T, N),
        torch.rand(N),
        0.99,
        0.95,
    )
    assert adv.shape == (T, N) and ret.shape == (T, N)
