# The guide: from the policy gradient to an asymmetric clip

This is the math I wish someone had put in one place. It goes from the plain
policy gradient to PPO's clipped objective, then splits the clip in two and
explains why the two halves might be doing different jobs to your policy's
entropy. Nothing here needs more than basic calculus and probability.

## 1. The policy gradient

We want a policy $\pi_\theta(a \mid s)$ that maximizes expected return
$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. The policy gradient
theorem says

$$\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \, A^{\pi}(s,a) \right]$$

where $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$ is the advantage: how much
better action $a$ is than the policy's average from state $s$. Positive
advantage → push the action's probability up; negative → push it down. We
estimate $A$ with GAE (section 4).

The catch: the expectation is under $\pi_\theta$ itself. The moment you take
one gradient step, your data was collected by the *wrong* policy, and the
estimator is stale. Vanilla policy gradient therefore takes exactly one step
per batch of experience — brutally sample-inefficient.

## 2. Importance sampling and the surrogate

To reuse a batch collected by an older policy $\pi_{\theta_{old}}$, reweight it:

$$L^{IS}(\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_{old}}}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \, A_t \right] = \mathbb{E}\left[ r_t(\theta) A_t \right]$$

The ratio $r_t(\theta)$ is 1 at the first gradient step and drifts as $\theta$
moves. The reweighting is only valid while $\pi_\theta$ stays close to
$\pi_{\theta_{old}}$ — far away, the variance explodes and the surrogate stops
tracking $J$. TRPO enforced closeness with an explicit KL constraint and a
conjugate-gradient solver. PPO replaced all of that with a `clamp`:

$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\!\big( r_t A_t,\; \mathrm{clip}(r_t,\, 1-\epsilon,\, 1+\epsilon) \, A_t \big) \right]$$

## 3. What the clip actually does (per sample)

Work through the four cases. The gradient of $L^{CLIP}$ w.r.t. a sample is
**zero** exactly when the min selects the clipped term:

| | $r_t < 1-\epsilon$ | $r_t \in [1-\epsilon, 1+\epsilon]$ | $r_t > 1+\epsilon$ |
|---|---|---|---|
| $A_t > 0$ | gradient **alive** | gradient alive | gradient **dead** |
| $A_t < 0$ | gradient **dead** | gradient alive | gradient **alive** |

Read the dead cells:

- **Top-right** ($A>0$, ratio already high): the policy already boosted this
  action a lot this update. The clip says *stop boosting*.
- **Bottom-left** ($A<0$, ratio already low): the policy already suppressed
  this action a lot. The clip says *stop suppressing*.

The alive-on-the-"wrong"-side cells (bottom-right, top-left) are the min's
pessimism: if the policy accidentally moved an action the wrong way, the
gradient stays on to pull it back. Clipping is one-sided per advantage sign.
`tests/test_clip.py` checks every cell of this table numerically.

## 4. GAE in three lines

The TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is a one-step
advantage estimate: low variance, biased by the critic. The Monte-Carlo
advantage is unbiased, high variance. GAE interpolates with a decay $\lambda$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}$$

computed backwards in one pass: $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$,
resetting at episode boundaries. $\lambda = 0$ is pure TD, $\lambda = 1$ is pure
Monte Carlo; 0.95 is the folk value nobody argues with. One subtlety this repo
handles that many implementations skip: when an episode is **truncated** (time
limit) rather than terminated, the environment didn't actually end — so we fold
$\gamma V(s_{final})$ into the last reward before treating it as a boundary.
`tests/test_gae.py` pins the recursion to hand-computed values.

## 5. Splitting the clip

Nothing forces the two bounds to share one $\epsilon$:

$$L(\theta) = \mathbb{E}\left[ \min\!\big( r_t A_t,\; \mathrm{clip}(r_t,\, 1-\epsilon_{low},\, 1+\epsilon_{high}) \, A_t \big) \right]$$

$\epsilon_{low} = \epsilon_{high} = 0.2$ is exactly PPO. Now map the bounds to
the dead cells of the table:

The key to both directions is asking **which actions actually reach each
bound**. The ratio of an action with old probability $p$ can never exceed
$1/p$ (probabilities cap at 1), and its log-prob gradient scales like
$(1-p)$. A dominant action at $p=0.5$ can't even reach ratio 2 and moves
slowly; a rare action at $p=0.01$ has a ratio ceiling of 100 and moves fast.
**Both clip bounds therefore bind almost exclusively on rare actions.** Then:

- $\epsilon_{low}$ controls the **bottom-left** dead zone — how long the
  policy may keep suppressing a rare, bad-looking action. Suppression drains
  tail probability mass toward zero. A *tight* $\epsilon_{low}$ halts the
  drain early → tail mass survives → entropy is held **up**.
- $\epsilon_{high}$ controls the **top-right** dead zone — how long the policy
  may keep boosting a rare, good-looking action. Boosting rare actions
  *rebuilds* tail mass. A *tight* $\epsilon_{high}$ freezes them almost
  immediately (their ratios blow past the bound in one step) → the tail never
  grows back → entropy is pushed **down**. Loosening it lets rare winners
  climb → entropy **up**.

That is the mechanism Park et al. ([arXiv:2509.26114](https://arxiv.org/abs/2509.26114))
report in RL fine-tuning of LLMs — clipping at the low bound preserves
entropy, clipping at the high bound destroys it — and DAPO
([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) exploits: its
"clip-higher" trick loosens $\epsilon_{high}$ precisely to fight entropy
collapse in frontier reasoning models. If the mechanism is real and general,
PPO's symmetric $\epsilon = 0.2$ has been quietly acting as an entropy
regulator since 2017, and the entropy bonus everyone tunes on top is partly
redundant.

(Confession, kept because it's instructive: my first draft of this section
argued the opposite sign for $\epsilon_{high}$ — "a loose upper bound lets
winners keep winning, so entropy falls." The pilot data said otherwise, and
the ratio-ceiling argument above explains why: the upper bound never binds on
winners, only on the tail. If your mental model of clipping is about the
dominant action, it's backwards.)

But every experiment in that literature has an action space of ~100k tokens.
DPPO ([arXiv:2602.04879](https://arxiv.org/abs/2602.04879)) explicitly argues
per-sample ratio clipping behaves qualitatively differently under huge
vocabularies. So the question this repo runs: **does the entropy mechanism
exist at $|A| = 6$, in the environments PPO was born in?** Either answer is
informative — transfer means the mechanism is fundamental to the objective;
no transfer means the hottest clipping trick in LLM-RL is a large-vocabulary
artifact.

## 6. Experimental design decisions (and why)

- **`ent_coef = 0` everywhere.** The entropy bonus is a confound: if it's on,
  entropy differences between clip settings could be the bonus interacting
  with the clip, not the clip itself. Off means every nat of entropy
  difference is attributable to clipping.
- **The clip must actually engage.** With the CleanRL-default
  `epochs=4, lr=2.5e-4` on MinAtar, we measured the ratio leaving the clip
  range on <0.3% of samples — a clip that never fires can't move entropy,
  and the whole grid would measure noise. We probed for a regime where
  clipping binds and learning still works
  (150k-step probes on breakout, seed 1):

  | config | return | clipfrac_low | clipfrac_high |
  |---|---|---|---|
  | epochs 4, lr 2.5e-4 | 2.0 | 0.002 | 0.001 |
  | epochs 8, lr 2.5e-4 | 3.0 | 0.003 | 0.006 |
  | epochs 4, lr 1e-3 | 4.0 | 0.012 | 0.010 |
  | epochs 8, lr 5e-4 | 4.0 | 0.010 | 0.011 |
  | **epochs 8, lr 1e-3** | **5.0** | **0.021** | **0.021** |

  More epochs per batch = more within-update drift = more clipping, which is
  exactly the knob Andrychowicz et al. ([arXiv:2006.05990](https://arxiv.org/abs/2006.05990))
  identify. `epochs=8, lr=1e-3` learns fastest *and* clips the most, so the
  sweep runs there.
- **Entropy is measured on the behavior policy** — logged during rollout
  collection, before any update touches the network. Measuring it mid-update
  would conflate the mechanism with optimizer noise.
- **5 seeds, mean ± sd, no cherry-picking.** MinAtar's entropy ceiling is
  $\ln 6 \approx 1.79$ nats, so effects will be small in absolute terms; seed
  bands decide whether they're real. If the bands overlap everywhere, the
  honest conclusion is "no detectable transfer at $|A|=6$" — that's a result,
  not a failure.

## 7. Reading list, in dependency order

1. Sutton & Barto ch. 13 — policy gradient basics.
2. Schulman et al., *High-Dimensional Continuous Control Using GAE* — [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
3. Schulman et al., *Proximal Policy Optimization Algorithms* — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
4. Engstrom et al., *Implementation Matters in Deep Policy Gradients* — [arXiv:2005.12729](https://arxiv.org/abs/2005.12729) — the paper that made everyone suspicious of §3.
5. Andrychowicz et al., *What Matters in On-Policy RL?* — [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
6. Park et al., *Clip-Low Increases Entropy and Clip-High Decreases Entropy in RL of LLMs* — [arXiv:2509.26114](https://arxiv.org/abs/2509.26114) — the claim under test.
7. DAPO — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); DPPO — [arXiv:2602.04879](https://arxiv.org/abs/2602.04879) — why the answer matters either way.
