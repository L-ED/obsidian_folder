#bfm #rl #fb-cpr

Losses of FB-CPR-AUX (BFM-Zero) as implemented in bfm1 (mjlab + rsl_rl, Unitree G1). Values are the G1 config.

Networks: forward map $F$, backward map $B$, discriminator $D$, discriminator critic $Q_{disc}$, aux critic $Q_{aux}$ (AUX only), actor $\pi$.

Notation:
- batch $n = 1024$, ensemble of $E = 2$ heads for every $F$/$Q$, $d = \dim z = 256$, $\gamma = 0.98$
- $\bar F, \bar B, \bar Q$ are target networks (EMA, $\tau = 0.01$ for $F, B$; $0.005$ for critics)
- $a' \sim \pi(s', z)$, std 0.05, clipped to 0.3
- pessimistic ensemble value, with $p$ the pessimism penalty:
$$
\mathrm{pess}(X) = \frac{1}{E}\sum_e X_e - p \cdot \frac{1}{E(E-1)}\sum_{i \neq j} |X_i - X_j|
$$

One gradient step (16 per env step): $D \rightarrow F, B \rightarrow Q_{disc}, Q_{aux} \rightarrow \pi \rightarrow$ target updates.

# 1. FB loss ($F$ and $B$)

Successor measure $M(s, a, s') = F(s, a, z)^\top B(s')$. Over a batch it is an $n \times n$ matrix
$$
M_e[i, j] = F_e(s_i, a_i, z_i) \cdot B(s'_j)
$$
Target (FB pessimism is 0, so it is the ensemble mean):
$$
\bar M = \mathrm{pess}\left(\bar F(s', a', z)\, \bar B(s')^\top\right), \qquad \Delta = M - \gamma \bar M
$$
$$
L_{offdiag} = \frac{1}{2} \sum_e \underset{i \neq j}{\mathrm{mean}}\ \Delta_e[i,j]^2
\qquad
L_{diag} = -\sum_e \underset{i}{\mathrm{mean}}\ \Delta_e[i,i]
$$
$$
L_{FB} = L_{offdiag} + L_{diag} + \lambda_{orth} L_{orth}
$$
Monte Carlo estimate of the Bellman equation of the successor measure:
- diagonal $(i, i)$: $s'_i$ is the real next state of $(s_i, a_i)$, so $M$ is pushed up there (the one-step mass)
- off-diagonal $(i \neq j)$: $s'_j$ is an independent sample of the data distribution $\rho$, so $M$ is pulled to the bootstrap $\gamma \bar M$

Together this minimizes $\frac{1}{2}\mathbb{E}_{s' \sim \rho}[(M - \gamma \bar M)^2] - \mathbb{E}[M(s, a, s_{next})]$: TD learning of the occupancy density relative to $\rho$.

TensorBoard: `fb_loss`, `fb_diag`, `fb_offdiag`; diagnostics `M1`, `target_M`, `F1`, `B`, `B_norm`, `z_norm`.

## Orthonormality loss ($B$ only, $\lambda_{orth} = 100$)

Goal: $B$ orthonormal under the data distribution
$$
C = \mathbb{E}_{s \sim \rho}[B(s) B(s)^\top] \approx I
$$
$$
\frac{1}{2}\|C - I\|_F^2 = \frac{1}{2}\mathrm{tr}(C^2) - \mathrm{tr}(C) + \frac{d}{2},
\qquad \mathrm{tr}(C^2) = \mathbb{E}_{s \neq s'}[(B(s) \cdot B(s'))^2], \quad \mathrm{tr}(C) = \mathbb{E}[\|B(s)\|^2]
$$
With the batch Gram matrix $G = B B^\top$ ($n \times n$):
$$
L_{orth,diag} = -\underset{i}{\mathrm{mean}}\ G_{ii}
\qquad
L_{orth,offdiag} = \frac{1}{2}\underset{i \neq j}{\mathrm{mean}}\ G_{ij}^2
\qquad
L_{orth} = L_{orth,offdiag} + L_{orth,diag} = \frac{1}{2}\|C - I\|^2 - \frac{d}{2}
$$
Fixes the $F$/$B$ scale ambiguity ($F \rightarrow F A^{-\top}$, $B \rightarrow B A$) and stops $B$ from collapsing to a few dimensions.

Reading it with `norm=True` ($\|B\| = \sqrt{d}$):
- `orth_loss_diag` is constant $-d = -256$
- `orth_loss_offdiag` $\geq d/2 = 128$, equality when $C = I$; it grows when $B$ collapses (all $B$ equal: $d^2/2 = 32768$)
- effective number of used dimensions (participation ratio) $= \mathrm{tr}(C)^2/\mathrm{tr}(C^2) = 32768 / $ `orth_loss_offdiag`
- `orth_loss` ideal value is $-128$

## Q loss (off for G1)

Extra TD loss on $Q = F \cdot z$ with the implicit reward $B\, \mathrm{Cov}(B)^{-1} z$. `q_loss_coef = 0`, so `q_loss` is always 0.

$F$ and $B$ share $L_{FB}$ but have separate Adam optimizers: lr 3e-4 for $F$, 1e-5 for $B$.

# 2. Discriminator loss ($D$)

$D(s, z) = \sigma(\ell(s, z))$ separates
- expert pairs: motion frame $s_t$ with $z = $ projected mean of $B$ over the next 8 frames $t+1 \dots t+8$
- policy pairs: replay state $s$ with the $z$ the policy acted under when it collected $s$

$$
L_D = \mathbb{E}_{expert}[-\log \sigma(\ell)] + \mathbb{E}_{policy}[-\log(1 - \sigma(\ell))] + \lambda_{GP}\, \mathbb{E}_{\hat x}\left[(\|\nabla_{s, z}\, \ell(\hat x)\| - 1)^2\right]
$$
Binary cross-entropy plus a WGAN-style gradient penalty ($\lambda_{GP} = 10$) at random interpolations $\hat x$ between expert and policy pairs, which keeps $D$ smooth so its reward stays informative. lr 1e-5.

TensorBoard: `disc_loss`, `disc_expert_loss`, `disc_train_loss`, `disc_wgan_gp_loss`.

# 3. Discriminator critic loss ($Q_{disc}$)

Reward = log-odds of the discriminator (= its logit), high when the policy's $(s, z)$ looks like the expert's:
$$
r_{disc} = \log \frac{D(s, z)}{1 - D(s, z)}
$$
$$
y = r_{disc} + \gamma\, \mathrm{pess}\left(\bar Q_{disc}(s', a', z)\right), \quad p = 0.5
\qquad
L_{Q_{disc}} = \frac{1}{2}\sum_e \mathbb{E}\left[(Q_{disc,e}(s, a, z) - y)^2\right]
$$
Ensemble TD with a pessimistic target: the Q-function of "imitate the motion that $z$ encodes". lr 3e-4.

TensorBoard: `critic_loss`, `Q1`, `target_Q`, `mean_next_Q`, `unc_Q`, `mean_disc_reward`.

# 4. Aux critic loss ($Q_{aux}$, AUX only)

Same TD loss on the env reward stored in the replay buffer: the weighted sum of BFM-Zero's penalties

| Term | Weight |
| --- | --- |
| joint position limits (95% of range) | -10 |
| ankle roll | -4 |
| feet slippage | -2 |
| undesired contact (pelvis, hips, shoulders) | -1 |
| feet orientation | -0.4 |
| action rate | -0.1 |

The reward is divided by its running (EMA) std, not mean-centred, so its scale does not depend on the penalty magnitudes:
$$
\tilde r_{aux} = \frac{r_{aux}}{\mathrm{std}_{EMA}(r_{aux})}
\qquad
y = \tilde r_{aux} + \gamma\, \mathrm{pess}\left(\bar Q_{aux}(s', a', z)\right), \quad p = 0.5
$$
$$
L_{Q_{aux}} = \frac{1}{2}\sum_e \mathbb{E}\left[(Q_{aux,e}(s, a, z) - y)^2\right]
$$
lr 3e-4.

TensorBoard: `aux_critic_loss`, `auxQ1`, `target_auxQ`, `mean_next_auxQ`, `unc_auxQ`, `mean_aux_reward` (normalized), `mean_raw_aux_reward`. The per-term penalties are `Episode_Reward/*` (rewards, not losses).

# 5. Actor loss ($\pi$)

With $a \sim \pi(s, z)$ and actor pessimism $p = 0.5$:
$$
Q_{fb} = \mathrm{pess}(F(s, a, z) \cdot z) \quad \text{reach what } z \text{ describes (zero-shot RL)}
$$
$$
Q_{disc} = \mathrm{pess}(Q_{disc}(s, a, z)) \quad \text{look like the expert motion of } z
$$
$$
Q_{aux} = \mathrm{pess}(Q_{aux}(s, a, z)) \quad \text{avoid the penalties (AUX only)}
$$
$$
w = \mathrm{mean}|Q_{fb}| \ \text{(no gradient)}
$$
$$
L_\pi = -Q_{fb} - \alpha_{disc}\, w\, Q_{disc} - \alpha_{aux}\, w\, Q_{aux}, \qquad \alpha_{disc} = 0.05, \ \alpha_{aux} = 0.02
$$
Scaling by $w$ (`scale_reg = True`) makes the coefficients relative to the size of the FB value: the regularizers stay about 5% and 2% of the main objective during training. Plain FB-CPR is the same loss without the $Q_{aux}$ term. lr 3e-4.

TensorBoard: `actor_loss`, `Q_fb`, `Q_discriminator`, `Q_aux`.

# Which z each loss uses

Each replay transition's $z$ is replaced with probability `relabel_ratio = 0.8` by a mixed $z$:
- 20% $B$ of a random replay state (goal reaching)
- 60% expert window encodings (tracking)
- 20% random $z$ on the sphere of radius $\sqrt d$ (reward-like)

$F$, $B$, both critics and the actor train on these relabeled $z$. The discriminator uses each policy pair's original $z$, because it judges what the policy actually did under that $z$.

# Code (bfm1)

- FB + orthonormality + Q loss: `src/bfm1/agents/fb/agent.py`, `update_fb`
- discriminator, critic, actor, z mixing: `src/bfm1/agents/fb_cpr/agent.py`
- aux critic and AUX actor: `src/bfm1/agents/fb_cpr_aux/agent.py`
- aux reward terms: `src/bfm1/tasks/mdp/rewards.py`, weights in `src/bfm1/tasks/g1/env_cfgs.py`

Related: [[FB-CPR related work]], [[PPO]], [[RL Relevant Notes]]
