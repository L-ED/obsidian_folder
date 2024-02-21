Paper https://arxiv.org/pdf/1707.06347.pdf

Common used gradient estimator

$$
\hat{g} = \hat{E}[\nabla_\theta \log{\pi_\theta(a_t|s_t)}\hat{A_t}]
$$
where E - expectation means average between elements of batch
TRPO is maximizing ratio of probabilities distributions of new and old policy with constraint that KL divergence less than given threshold. Constrain keeping model from failing to far in bad domains.
$$
\max_{\theta} \hat{E_t}[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A_t}] \quad subject \: \hat{E_t}[KL(\pi_\theta(a_t|s_t),\pi_{\theta_{old}}(a_t|s_t))] \leq \delta
$$
PPO is updating policy with pessimistic bound, ignoring the probability ratio on improvements and including ratio in gradient when objective getting worse 
$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} 
$$
$$
L_t^{clip}=\hat{E_t}[min(r_t(\theta)\hat{A_t}, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A_t})]
$$
If using a neural network architecture that shares parameters between the policy and value function, we must use a loss function that combines the policy surrogate and a value function error term. Combining these terms, we obtain the following objective, which is (approximately) maximized each iteration
$$
L_t^{clip+vf+s}(\theta)=\hat{E_t}[L_t^{clip}-c_v*L_t^{value}+c_e*entropy[\pi_{\theta}(s_t)]]
$$
$$
L_t^{vf} = (V_{\theta}(s_t)-V_t^{targ})^2
$$
