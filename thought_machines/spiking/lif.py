"""Leaky integrate-and-fire neuron: the simplest spiking model.

    tau * dv/dt = -(v - v_rest) + R * I

The membrane voltage leaks toward rest, integrates input current, and when it
reaches threshold the neuron emits a spike, resets, and stays silent for a
refractory period. Information is carried by spike times, not by a continuous
activation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LIFParams:
    tau: float = 20.0      # membrane time constant, ms
    v_rest: float = -65.0  # mV
    v_reset: float = -65.0
    v_th: float = -50.0
    r: float = 1.0         # input resistance, mV per unit of current
    t_ref: float = 2.0     # refractory period, ms


def simulate_lif(current, t_max: float = 1000.0, dt: float = 0.1, params: LIFParams = LIFParams(),
                 record: bool = False):
    """Euler integration. `current` is a number or a function of time in ms.

    Returns (spike_times_ms, voltage_trace)."""
    p = params
    v, last = p.v_rest, -1e9
    spikes, trace = [], []
    for k in range(int(round(t_max / dt))):
        t = k * dt
        i = current(t) if callable(current) else current
        if t - last < p.t_ref:
            v = p.v_reset
        else:
            v += dt / p.tau * (-(v - p.v_rest) + p.r * i)
            if v >= p.v_th:
                spikes.append(t)
                v, last = p.v_reset, t
        if record:
            trace.append(v)
    return spikes, trace


def analytic_rate_hz(i: float, params: LIFParams = LIFParams()) -> float:
    """Firing rate for constant input, valid when v_reset equals v_rest."""
    p = params
    drive, gap = p.r * i, p.v_th - p.v_rest
    if drive <= gap:
        return 0.0
    isi = p.t_ref + p.tau * math.log(drive / (drive - gap))
    return 1000.0 / isi


def rate_hz(spikes, t_max: float) -> float:
    return 1000.0 * len(spikes) / t_max
