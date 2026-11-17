"""Steering: move toward what is good and away from what is bad.

Bennett's first breakthrough, in early bilaterians, is steering. An animal with no
map and no memory beyond the last instant can still find food if it compares "better
or worse than a moment ago" and acts on the answer. Two classic models are here.

* Run-and-tumble, the way bacteria climb a chemical gradient. Keep going straight
  while the signal improves, and reorient at random when it does not. A `valence`
  of +1 approaches the source and -1 avoids it, which is the whole idea of good and
  bad stimuli at its smallest.
* Braitenberg vehicles, two sensors wired to two wheels. Same-side wiring turns the
  vehicle away from a source, crossed wiring turns it toward the source. No
  representation of the source exists anywhere, only wiring.

Everything is seeded and deterministic for a given seed.
"""

from __future__ import annotations

import math
import random

SOURCE = (0.0, 0.0)


def concentration(x: float, y: float, source=SOURCE) -> float:
    """A smooth attractant field with its peak at the source."""
    return 1.0 / (1.0 + math.hypot(x - source[0], y - source[1]))


class RunAndTumble:
    """Memory of one number: the signal a moment ago."""

    def __init__(self, p_better: float = 0.02, p_worse: float = 0.5, valence: int = 1, seed: int = 0):
        if valence not in (1, -1):
            raise ValueError("valence must be +1 (approach) or -1 (avoid)")
        self.p_better, self.p_worse, self.valence = p_better, p_worse, valence
        self.rng = random.Random(seed)

    def tumble_probability(self, change: float) -> float:
        return self.p_better if self.valence * change > 0 else self.p_worse

    def turn(self, heading: float, change: float) -> float:
        if self.rng.random() < self.tumble_probability(change):
            return self.rng.uniform(0, 2 * math.pi)
        return heading


def chemotaxis_trial(agent: RunAndTumble, start=(20.0, 0.0), steps: int = 400, speed: float = 1.0):
    """Run one agent. Returns the list of distances to the source over time."""
    x, y = start
    heading = agent.rng.uniform(0, 2 * math.pi)
    previous = concentration(x, y)
    distances = [math.hypot(x, y)]
    for _ in range(steps):
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)
        now = concentration(x, y)
        heading = agent.turn(heading, now - previous)
        previous = now
        distances.append(math.hypot(x, y))
    return distances


def random_walk_trial(seed: int, rate: float, start=(20.0, 0.0), steps: int = 400, speed: float = 1.0):
    """The control: same movement, but the tumble rate ignores the signal."""
    return chemotaxis_trial(RunAndTumble(p_better=rate, p_worse=rate, seed=seed), start, steps, speed)


def braitenberg_run(crossed: bool, start=(10.0, 0.0), heading: float = math.pi / 2, steps: int = 300,
                    gain: float = 10.0, base: float = 0.15, width: float = 0.4, dt: float = 0.5,
                    sensor_angle: float = math.pi / 6, sensor_reach: float = 1.0,
                    stop_radius: float = 1.5, source=SOURCE):
    """A two-wheeled vehicle. Returns the list of (x, y) positions.

    crossed=True wires the left sensor to the right wheel (turns toward the source);
    crossed=False wires each sensor to the wheel on its own side (turns away). The run
    ends early if the vehicle gets within `stop_radius` of the source (it has arrived).
    """
    x, y = start
    path = [(x, y)]
    for _ in range(steps):
        left = concentration(x + sensor_reach * math.cos(heading + sensor_angle),
                             y + sensor_reach * math.sin(heading + sensor_angle), source)
        right = concentration(x + sensor_reach * math.cos(heading - sensor_angle),
                              y + sensor_reach * math.sin(heading - sensor_angle), source)
        v_left = base + gain * (right if crossed else left)
        v_right = base + gain * (left if crossed else right)
        heading += dt * (v_right - v_left) / width
        speed = (v_left + v_right) / 2
        x += dt * speed * math.cos(heading)
        y += dt * speed * math.sin(heading)
        path.append((x, y))
        if math.hypot(x - source[0], y - source[1]) <= stop_radius:
            break
    return path
