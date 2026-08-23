# The guide

This is the document I wish someone had handed me before PPO. It goes from
"what problem is RL even solving" to the full training loop, then splits the
clip range in two and walks through what we found. Nothing here needs more
than basic calculus and probability. Read it next to `ppo/core.py` — every
piece of math below names the function that implements it.

**Already know PPO?** Skip to [§7](#7-splitting-the-clip) for the asymmetric
clip and [§9](#9-what-the-experiment-found) for the results.

## 0. The problem

An agent sees a state $s_t$, picks an action $a_t$, gets a reward $r_t$, and
the environment moves to $s_{t+1}$. A **policy** $\pi_\theta(a \mid s)$ is a
neural net that outputs a probability distribution over actions. We want the
$\theta$ that maximizes expected discounted return

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \gamma^t r_t \right]$$

Two helper quantities show up everywhere:

- $V^\pi(s)$ — the **value** of a state: expected return from $s$ if you
  follow $\pi$. In this repo it's the critic head in `core.py`.
- $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ — the **advantage**: how much better
  taking $a$ from $s$ is than whatever $\pi$ would do on average. Positive =
  better than usual, negative = worse. Advantages, not raw rewards, are what
  drive every gradient in PPO.

## 1. The policy gradient

The policy gradient theorem says

$$\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \, A^{\pi}(s,a) \right]$$

Read it as an instruction: *increase the log-probability of actions in
proportion to how much better than average they were.* Positive advantage →
push the action up; negative → push it down.

The catch is in the subscript: the expectation is under $\pi_\theta$
*itself*. Take one gradient step and your data was collected by the wrong
policy — the estimator is stale. Vanilla policy gradient therefore takes
exactly one step per batch of experience, which is brutally sample-hungry.

## 2. Reusing data: importance sampling

To squeeze several gradient steps out of one batch collected by an older
policy $\pi_{\theta_{old}}$, reweight each sample by how much more (or less)
likely the new policy is to produce it:

$$L^{IS}(\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_{old}}}\left[ r_t(\theta) \, A_t \right], \qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

The **ratio** $r_t$ is exactly 1 at the first gradient step and drifts as
$\theta$ moves. The reweighting is only valid while the new policy stays
close to the old one — far away, variance explodes and the surrogate stops
tracking $J$. TRPO enforced closeness with an explicit KL constraint and a
second-order solver. PPO replaced all of that machinery with a `clamp`:

$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\!\big( r_t A_t,\; \mathrm{clip}(r_t,\, 1-\epsilon,\, 1+\epsilon) \, A_t \big) \right]$$

Implemented in `ppo_policy_loss` in `ppo/core.py` — four lines.

## 3. What the clip actually does, per sample

Work through the four cases. The gradient of $L^{CLIP}$ w.r.t. a sample is
**zero** exactly when the min selects the clipped term:

| | $r_t < 1-\epsilon$ | $r_t \in [1-\epsilon,\, 1+\epsilon]$ | $r_t > 1+\epsilon$ |
|---|---|---|---|
| $A_t > 0$ | gradient **alive** | gradient alive | gradient **dead** |
| $A_t < 0$ | gradient **dead** | gradient alive | gradient **alive** |

Read the dead cells:

- **Top-right** ($A>0$, ratio already high): the policy already boosted this
  action a lot this update. The clip says *stop boosting*.
- **Bottom-left** ($A<0$, ratio already low): the policy already suppressed
  this action a lot. The clip says *stop suppressing*.

The alive-on-the-"wrong"-side cells are the min's pessimism: if the policy
accidentally moved an action the wrong way, the gradient stays on to pull it
back. Clipping is one-sided per advantage sign. `tests/test_clip.py` checks
every cell of this table numerically — read it next to this section.

## 4. The critic and GAE

The policy gradient needs advantages, and advantages need a baseline. The
critic learns $V(s)$ by regressing onto observed returns (the value loss in
the update loop). Given a critic, the one-step TD error

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

is already an advantage estimate: low variance, but biased wherever the
critic is wrong. The Monte-Carlo alternative (sum all future rewards, subtract
$V(s_t)$) is unbiased but noisy. **GAE** interpolates with a decay $\lambda$:

$$\hat{A}_t = \sum_{l\ge 0} (\gamma\lambda)^l \, \delta_{t+l} \quad\Longleftrightarrow\quad \hat{A}_t = \delta_t + \gamma\lambda\,\hat{A}_{t+1}$$

$\lambda=0$ is pure TD, $\lambda=1$ is pure Monte Carlo; 0.95 is the folk
value nobody argues with. Implemented as the backwards loop in `compute_gae`.

A worked example you can check by hand (it's literally
`tests/test_gae.py::test_gae_recursion_by_hand`, with $\gamma=\lambda=0.5$):
rewards $[0, 1]$, values $[1, 1]$, bootstrap $V(s_2)=4$. Then
$\delta_1 = 1 + 0.5\cdot4 - 1 = 2$, $\delta_0 = 0 + 0.5\cdot1 - 1 = -0.5$,
so $\hat A_1 = 2$ and $\hat A_0 = -0.5 + 0.25\cdot2 = 0$.

One subtlety this repo handles that many single-file PPOs skip: when an
episode is **truncated** (time limit) rather than terminated, the environment
didn't actually end, so we fold $\gamma V(s_{final})$ into the last reward
before treating the step as a boundary. Skipping this biases the critic on
every truncated episode.

## 5. Anatomy of one PPO iteration

The whole of `train` in `ppo/core.py` is this loop:

1. **Collect** `rollout_len × num_envs` steps with the current policy,
   storing observations, actions, log-probs, rewards, values — and the
   policy's entropy at each step, *before* any update touches the network
   (that's the entropy every figure in this repo plots).
2. **Bootstrap + GAE** — compute $\hat A_t$ and value targets.
3. **Update** for `epochs` passes over the batch, in shuffled minibatches:
   - recompute log-probs under the *current* $\theta$, form ratios against
     the stored collection-time log-probs;
   - normalize advantages per minibatch (mean 0, std 1 — an empirical
     stabilizer, one of the "code-level optimizations" Engstrom et al. showed
     matter more than people admit);
   - policy loss = the clipped surrogate from §3;
   - value loss = MSE to the targets (optionally clipped — a folk trick kept
     here for CleanRL comparability, fixed at 0.2 and *not* tied to the
     policy clip, so the experiment's knobs stay clean);
   - subtract `ent_coef ×` entropy (an explicit exploration bonus — **zero in
     every experiment here**, which is the whole point);
   - clip the global gradient norm, Adam step.
4. Linearly anneal the learning rate over training.

More epochs per batch = more within-update drift = larger ratios = the clip
engages more. That knob matters in §8.

## 6. Reading the telemetry

Every update logs a JSON line (`metrics.jsonl`). What the fields mean and
what healthy looks like:

- `entropy` — behavior-policy entropy, in nats, measured at collection time.
  Uniform over 6 actions = $\ln 6 \approx 1.79$. Slow decline is normal;
  a crash to ~0 means the policy went deterministic (possibly prematurely).
- `approx_kl` — the k3 estimator $\mathbb{E}[(r-1) - \log r]$ of
  $KL(\pi_{old} \| \pi)$ per minibatch. Rule of thumb: consistently above
  ~0.02 means updates are too hot.
- `clipfrac_low` / `clipfrac_high` — fraction of samples whose ratio sits
  below $1-\epsilon_{low}$ / above $1+\epsilon_{high}$. If both are ~0, the
  clip is decorative and *any* clip experiment on that config measures noise
  (§8 exists because of this).
- `episodic_return` — rolling mean of the last 100 finished episodes.

## 7. Splitting the clip

Nothing forces the two bounds to share one $\epsilon$:

$$L(\theta) = \mathbb{E}\left[ \min\!\big( r_t A_t,\; \mathrm{clip}(r_t,\, 1-\epsilon_{low},\, 1+\epsilon_{high}) \, A_t \big) \right]$$

$\epsilon_{low} = \epsilon_{high} = 0.2$ is exactly PPO.

The key to what the bounds do is asking **which actions actually reach
them**. The ratio of an action with old probability $p$ can never exceed
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

Every experiment in that literature has an action space of ~100k tokens, and
DPPO ([arXiv:2602.04879](https://arxiv.org/abs/2602.04879)) explicitly argues
per-sample ratio clipping behaves qualitatively differently under huge
vocabularies. Hence this repo's question: **does the mechanism exist at
$|A| = 6$, in the environments PPO was born in?**

(Confession, kept because it's instructive: my first draft of this section
argued the opposite sign for $\epsilon_{high}$ — "a loose upper bound lets
winners keep winning, so entropy falls." The pilot data said otherwise, and
the ratio-ceiling argument above explains why: the upper bound never binds on
winners, only on the tail. If your mental model of clipping is about the
dominant action, it's backwards.)

## 8. Experimental design decisions (and why)

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
  collection, before any update touches the network.
- **5 seeds, mean ± sd, no cherry-picking.** MinAtar's entropy ceiling is
  $\ln 6 \approx 1.79$ nats, so effects are small in absolute terms; seed
  bands decide whether they're real.

## 9. What the experiment found

Full grid: `clip_low` ∈ {0.05, 0.1, 0.2, 0.3} × `clip_high` ∈ {0.1, 0.2,
0.3, 0.5} × 5 seeds × 3M steps, on breakout and asterix. Figures in the
README and `figures/`.

**Finding 1 — the mechanism transfers.** Final entropy is monotone along both
axes of both games' heatmaps: tighter $\epsilon_{low}$ → higher entropy,
looser $\epsilon_{high}$ → higher entropy. On breakout the clip setting alone
spans 0.30 to 1.23 nats. Park et al.'s signs, reproduced 4 orders of
magnitude below the action-space size they were discovered at. This is
evidence *against* the strong version of DPPO's claim that the clipping
mechanism is a large-vocabulary artifact.

**Finding 2 — $\epsilon_{low}$ is the dominant dial**, moving entropy roughly
3× further than $\epsilon_{high}$ at matched widths. A plausible reading of
the asymmetry: suppression (what $\epsilon_{low}$ limits) compounds every
update an action stays out of favor, while boosting (what $\epsilon_{high}$
limits) only fires in the rarer moments a tail action looks good. Untested
here — measuring *who* occupies each dead zone over training would settle it.

**Finding 3 — the performance effect does not transfer; it flips.** High
entropy wins breakout (21.4 best return vs ~9 for the low-entropy row) and
loses asterix (best cell 19.1 is low-entropy; the max-entropy corner scores
7.6). A story consistent with this — offered as hypothesis, not conclusion:
in breakout, exploratory paddle movement finds new ball angles and randomness
is cheap; in asterix, enemies kill you, so residual stochasticity is a tax on
every episode. The general lesson is the honest one: **clip asymmetry is a
real entropy knob, but the right entropy is a property of the environment.**

What this repo does *not* show: whether the mechanism scales with action-space
size (the 6 → 18 → hundreds dose-response that would fully arbitrate DPPO's
hypothesis), whether it holds at other `epochs`/`lr` regimes, or whether tuned
asymmetry can replace the entropy bonus at matched final performance. All
three are natural next chapters.

## 10. Reading list, in dependency order

1. Sutton & Barto ch. 13 — policy gradient basics.
2. Schulman et al., *High-Dimensional Continuous Control Using GAE* — [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
3. Schulman et al., *Proximal Policy Optimization Algorithms* — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
4. Engstrom et al., *Implementation Matters in Deep Policy Gradients* — [arXiv:2005.12729](https://arxiv.org/abs/2005.12729) — the paper that made everyone suspicious of §3.
5. Andrychowicz et al., *What Matters in On-Policy RL?* — [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
6. Park et al., *Clip-Low Increases Entropy and Clip-High Decreases Entropy in RL of LLMs* — [arXiv:2509.26114](https://arxiv.org/abs/2509.26114) — the claim under test.
7. DAPO — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); DPPO — [arXiv:2602.04879](https://arxiv.org/abs/2602.04879) — why the answer matters either way.
