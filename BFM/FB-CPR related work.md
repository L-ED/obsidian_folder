#bfm #rl #fb-cpr #papers

Recent (2025–2026) papers that modify or build on FB-CPR / forward-backward (FB) representations. Collected Nov 2026 from abstracts and summaries, so the reported numbers are the authors' claims, not checked against the full texts.

See also [[BFM-Zero losses]] for the losses of the method these build on.

# Lineage

- **FB** (Touati & Ollivier): successor measure $M(s, a, s') = F(s, a, z)^\top B(s')$, zero-shot RL by $z = \mathbb{E}[r(s) B(s)]$
- **FB-CPR / Meta Motivo** ([arXiv 2504.11054](https://arxiv.org/abs/2504.11054), ICLR 2025): FB plus conditional policy regularization, i.e. a latent-conditioned discriminator on unlabeled motions (expert $z$ = B-encoding of the motion window) and a critic on the discriminator reward. Ablations: the latent-conditioned discriminator beats a state-only one on tracking. Weak on behaviors far from the mocap data (e.g. ground motions). Code: [facebookresearch/metamotivo](https://github.com/facebookresearch/metamotivo)
- **BFM-Zero** ([arXiv 2511.04131](https://arxiv.org/abs/2511.04131), ICLR 2026, [project](https://lecar-lab.github.io/BFM-Zero/)): FB-CPR for a real Unitree G1. Adds the aux critic on penalty rewards (FB-CPR-AUX), domain randomization, history-dependent asymmetric observations (privileged critic). Zero-shot tracking, goal reaching, reward inference, few-shot adaptation. **Base of bfm1.**

# Humanoid BFM papers

| Paper | Change compared to FB-CPR / BFM-Zero |
| --- | --- |
| **P-BFM** ([2610.15833](https://github.com/Roboparty/P-BFM/wiki), Oct 2026) | Perception. Splits control into terrain-invariant intent (FB latent) and terrain-conditioned execution from depth. **Explicit Root Instructions**: heading and planar velocity given explicitly, not only through $z$, because the latent loses precise global progress. Zero-shot on G1 with onboard depth. |
| **Perceptive BFM** ([2606.08059](https://arxiv.org/abs/2606.08059), Jun 2026) | Adapts human motion priors to terrain with a teacher–student setup and residual correction. Tracking-based, not FB, as far as the abstract shows. |
| **Belief-FB / Rotation-FB** ([2505.13150](https://arxiv.org/abs/2505.13150), ICLR 2026) | FB cannot tell different dynamics apart, so latent directions interfere. Belief-FB: transformer belief estimator of the dynamics. Rotation-FB: splits $z$-space into one cluster per dynamics. Up to 2x zero-shot return under changed dynamics. |
| **Scaling BFM for Humanoids** ([2607.15163](https://arxiv.org/abs/2607.15163), Jul 2026) | Not FB (cites BFM-Zero as prior work): global-frame tracking with a Transformer. Data diversity matters more than amount; scale width and depth together. |

# Changes to the FB core

| Paper | Idea |
| --- | --- |
| **One-step FB**, "Can We Really Learn One Representation to Optimize All Rewards?" ([2602.11399](https://arxiv.org/abs/2602.11399), ICML 2026) | The FB objective is circular (representations and policy depend on each other). Learns FB of a *fixed behavior policy* plus one policy-improvement step. ~$10^5$x lower representation error, +24% zero-shot. Sparse goal rewards may collapse the diversity of $B$. |
| **RLDP** ([2603.15857](https://arxiv.org/abs/2603.15857), ICLR 2026) | Strong simple baseline: latent next-state prediction plus an **orthogonality regularizer** against feature collapse. Matches or beats FB-style methods, more robust with low data coverage. |
| **BREEZE**, "Towards Robust Zero-Shot RL" ([2510.15382](https://arxiv.org/abs/2510.15382)) | Offline FB with behavior regularization (against out-of-distribution actions), more expressive F/B, task-conditioned **diffusion policy**. |
| **Soft FB** ([2602.06769](https://arxiv.org/abs/2602.06769), Feb 2026) | Maximum-entropy FB with stochastic policies. Optimizes general utilities (any differentiable function of the occupancy), not only linear rewards. |
| **Zero-Shot Off-Policy Learning** ([2602.01962](https://arxiv.org/abs/2602.01962)) | FB encodes density ratios between task policy and data; uses them for training-free search over $z$ at test time. |
| **Optimistic task inference** ([2510.20264](https://arxiv.org/abs/2510.20264)) | Better reward → $z$ inference from few labeled samples, optimism over uncertainty. |
| **FB-AWARE**, "Finer BFMs" ([2412.04368](https://arxiv.org/abs/2412.04368)) | Auto-regressive features (beyond linear rewards) plus advantage-weighted fine-tuning; humanoid tasks. |
| **DVFB** (ICLR 2025) | Exploration value and downstream value functions so online FB can improve beyond zero-shot. |
| Epistemically-guided FB exploration (RLC 2025) | Exploration as minimizing the epistemic uncertainty of the value functions. |

# Ideas for bfm1

1. **Explicit root commands** (P-BFM): addresses FB-CPR's root drift in tracking; check with `error_anchor_pos` / `error_anchor_rot`.
2. **Dynamics-aware $z$** (Belief-FB / Rotation-FB): bfm1 trains under domain randomization, where one $z$ must work for all dynamics.
3. **One-step FB**: stabilizes the FB loss; deeper change, but targets representation error directly.
4. **Watch $B$ collapse** (RLDP, one-step FB): effective dims of $B$ $= 32768 /$ `orth_loss_offdiag` (for $d = 256$, normalized $B$), cheap to add to TensorBoard.
