"""Spike-timing-dependent plasticity: learning from the order of spikes.

If a presynaptic spike arrives shortly before the postsynaptic neuron fires, the
synapse is strengthened. If it arrives shortly after, the synapse is weakened.
Applied to a neuron driven by a mix of correlated and uncorrelated inputs, this
simple rule picks out the inputs that tend to take part in making it fire.
"""

from __future__ import annotations

import math
import random


def stdp_window(delta_t: float, a_plus: float = 0.01, a_minus: float = 0.0105,
                tau_plus: float = 20.0, tau_minus: float = 20.0) -> float:
    """Weight change for delta_t = t_post - t_pre (ms)."""
    if delta_t > 0:
        return a_plus * math.exp(-delta_t / tau_plus)
    if delta_t < 0:
        return -a_minus * math.exp(delta_t / tau_minus)
    return 0.0


def simulate_stdp(n_inputs: int = 40, n_correlated: int = 20, seconds: float = 100.0, seed: int = 0,
                  base_rate: float = 5.0, volley_rate: float = 5.0, w_max: float = 1.0,
                  a_plus: float = 0.01, a_minus: float = 0.0105, tau: float = 20.0,
                  threshold: float = 10.0, jump: float = 1.0):
    """Returns the final weights. The first `n_correlated` inputs share common volleys.

    Every input fires at about base_rate + volley_rate Hz when correlated, or
    base_rate + volley_rate Hz independently otherwise, so rates alone carry no signal."""
    rng = random.Random(seed)
    steps = int(seconds * 1000)
    p_ind_corr = base_rate / 1000.0
    p_ind_unc = (base_rate + volley_rate) / 1000.0
    p_volley = volley_rate / 1000.0
    w = [0.5 * w_max] * n_inputs
    pre_trace = [0.0] * n_inputs
    post_trace = 0.0
    v = 0.0
    decay_v = math.exp(-1.0 / tau)
    decay_tr = math.exp(-1.0 / tau)
    for _ in range(steps):
        volley = rng.random() < p_volley
        v *= decay_v
        pre_trace = [x * decay_tr for x in pre_trace]
        post_trace *= decay_tr
        fired = []
        for i in range(n_inputs):
            if i < n_correlated:
                s = volley or rng.random() < p_ind_corr
            else:
                s = rng.random() < p_ind_unc
            if s:
                fired.append(i)
        for i in fired:
            v += jump * w[i]
            pre_trace[i] += 1.0
            w[i] = max(0.0, w[i] - a_minus * w_max * post_trace)  # pre after post: depress
        if v >= threshold:
            v = 0.0
            post_trace += 1.0
            for i in range(n_inputs):
                w[i] = min(w_max, w[i] + a_plus * w_max * pre_trace[i])  # pre before post: potentiate
    return w
