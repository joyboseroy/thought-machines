import unittest
from itertools import product

from thought_machines.computation import circuits as c
from thought_machines.computation.turing import (binary_increment, busy_beaver_3, palindrome,
                                                  unary_add)


class CircuitTests(unittest.TestCase):
    def test_nand_built_gates_match_python(self):
        for a, b in product((0, 1), repeat=2):
            self.assertEqual(c.and_(a, b), a & b)
            self.assertEqual(c.or_(a, b), a | b)
            self.assertEqual(c.xor_(a, b), a ^ b)
            self.assertEqual(c.nor_(a, b), 1 - (a | b))
        self.assertEqual([c.not_(0), c.not_(1)], [1, 0])

    def test_ripple_adder_is_exhaustively_correct_on_4_bits(self):
        for x, y in product(range(16), repeat=2):
            out = c.from_bits(c.ripple_add(c.to_bits(x, 4), c.to_bits(y, 4)))
            self.assertEqual(out, x + y)


class TuringTests(unittest.TestCase):
    def test_increment(self):
        tm = binary_increment()
        for n in range(64):
            self.assertEqual(int(tm.run(bin(n)[2:]).tape, 2), n + 1)

    def test_unary_add(self):
        for a, b in product(range(1, 5), repeat=2):
            self.assertEqual(unary_add().run("1" * a + "+" + "1" * b).tape, "1" * (a + b))

    def test_palindrome(self):
        tm = palindrome()
        for w in ("", "a", "aa", "aba", "abba", "baab", "ababa"):
            self.assertEqual(tm.run(w).state, "accept", w)
        for w in ("ab", "abb", "abab", "aab", "baa"):
            self.assertEqual(tm.run(w).state, "reject", w)

    def test_busy_beaver_3(self):
        r = busy_beaver_3().run("")
        self.assertTrue(r.halted)
        self.assertEqual((r.ones, r.steps), (6, 14))

    def test_step_limit_stops_a_non_halting_run(self):
        from thought_machines.computation.turing import TuringMachine
        loop = TuringMachine({("s", "_"): ("_", "R", "s")}, start="s")
        r = loop.run("", max_steps=50)
        self.assertFalse(r.halted)
        self.assertEqual(r.steps, 50)


if __name__ == "__main__":
    unittest.main()
