"""Kanerva's sparse distributed memory.

A very large binary address space is covered by a modest number of randomly
placed "hard locations". Writing to an address updates every hard location within
a Hamming radius. Reading sums the counters of the locations near the cue and
thresholds. Because nearby addresses activate overlapping sets of locations, a
noisy cue still lands on mostly the right counters, and iterating the read cleans
it up further.
"""

from __future__ import annotations

import random


def popcount(x: int) -> int:
    return bin(x).count("1")


class SDM:
    def __init__(self, n_bits: int = 256, n_locations: int = 2000, radius: int = 112, seed: int = 0):
        self.n, self.radius = n_bits, radius
        self.rng = random.Random(seed)
        self.addresses = [self.rng.getrandbits(n_bits) for _ in range(n_locations)]
        self.counters = [[0] * n_bits for _ in range(n_locations)]

    def active(self, address: int):
        return [i for i, a in enumerate(self.addresses) if popcount(a ^ address) <= self.radius]

    def write(self, address: int, data: int):
        bits = [1 if (data >> j) & 1 else -1 for j in range(self.n)]
        for i in self.active(address):
            row = self.counters[i]
            for j, b in enumerate(bits):
                row[j] += b

    def read(self, address: int) -> int:
        total = [0] * self.n
        for i in self.active(address):
            for j, c in enumerate(self.counters[i]):
                total[j] += c
        return sum(1 << j for j, t in enumerate(total) if t > 0)

    def recall(self, cue: int, iterations: int = 6) -> int:
        """Iterated read: feed the output back in as the next cue until it stops changing."""
        x = cue
        for _ in range(iterations):
            y = self.read(x)
            if y == x:
                break
            x = y
        return x

    def random_word(self) -> int:
        return self.rng.getrandbits(self.n)

    def noisy(self, word: int, flips: int) -> int:
        for j in self.rng.sample(range(self.n), flips):
            word ^= 1 << j
        return word
