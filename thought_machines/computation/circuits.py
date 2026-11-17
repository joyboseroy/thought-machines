"""Boole to circuits: every gate built from NAND alone, then an adder.

The chain is: logical operations -> gates -> circuits -> arithmetic.
"""

from __future__ import annotations

from itertools import product


def nand(a: int, b: int) -> int:
    return 0 if (a and b) else 1


def not_(a: int) -> int:
    return nand(a, a)


def and_(a: int, b: int) -> int:
    return not_(nand(a, b))


def or_(a: int, b: int) -> int:
    return nand(not_(a), not_(b))


def xor_(a: int, b: int) -> int:
    n = nand(a, b)
    return nand(nand(a, n), nand(b, n))


def nor_(a: int, b: int) -> int:
    return not_(or_(a, b))


def half_adder(a: int, b: int):
    """Returns (sum, carry)."""
    return xor_(a, b), and_(a, b)


def full_adder(a: int, b: int, carry_in: int):
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, carry_in)
    return s2, or_(c1, c2)


def to_bits(n: int, width: int) -> list[int]:
    """Least significant bit first."""
    return [(n >> i) & 1 for i in range(width)]


def from_bits(bits) -> int:
    return sum(b << i for i, b in enumerate(bits))


def ripple_add(a_bits, b_bits):
    """Add two LSB-first bit lists with a chain of full adders. Output has one extra bit."""
    n = max(len(a_bits), len(b_bits))
    a = list(a_bits) + [0] * (n - len(a_bits))
    b = list(b_bits) + [0] * (n - len(b_bits))
    carry, out = 0, []
    for x, y in zip(a, b):
        s, carry = full_adder(x, y, carry)
        out.append(s)
    return out + [carry]


def truth_table(fn, arity: int = 2):
    return [(bits, fn(*bits)) for bits in product((0, 1), repeat=arity)]
