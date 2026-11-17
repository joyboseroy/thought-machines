"""Izhikevich neuron: two equations, many firing patterns.

    v' = 0.04 v^2 + 5 v + 140 - u + I
    u' = a (b v - u)         and on a spike (v >= 30 mV): v <- c, u <- u + d

Changing four numbers (a, b, c, d) switches between regular spiking, bursting,
chattering and fast spiking, which is why the model is popular for large network
simulations.
"""

from __future__ import annotations

PRESETS = {
    "regular spiking": (0.02, 0.2, -65.0, 8.0),
    "intrinsically bursting": (0.02, 0.2, -55.0, 4.0),
    "chattering": (0.02, 0.2, -50.0, 2.0),
    "fast spiking": (0.1, 0.2, -65.0, 2.0),
}


def simulate_izhikevich(a, b, c, d, current, t_max: float = 1000.0, dt: float = 1.0):
    """Returns spike times in ms. `current` is a number or a function of time."""
    v, u = -65.0, b * -65.0
    spikes = []
    for k in range(int(round(t_max / dt))):
        t = k * dt
        i = current(t) if callable(current) else current
        if v >= 30.0:
            spikes.append(t)
            v, u = c, u + d
        for _ in range(2):  # two half steps for v keeps the quadratic term stable
            v += 0.5 * dt * (0.04 * v * v + 5 * v + 140 - u + i)
        u += dt * a * (b * v - u)
    return spikes


def intervals(spikes):
    return [b - a for a, b in zip(spikes, spikes[1:])]


def cv(spikes) -> float:
    """Coefficient of variation of inter-spike intervals. Near 0 is clock-like, large is bursty."""
    isi = intervals(spikes)
    if len(isi) < 2:
        return 0.0
    m = sum(isi) / len(isi)
    var = sum((x - m) ** 2 for x in isi) / len(isi)
    return var ** 0.5 / m
