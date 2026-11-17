"""A Turing machine written from scratch, plus a few classic machines.

Rules map (state, symbol) -> (symbol to write, move, next state), with move in
{L, R, S}. The tape is a dict, so it is unbounded in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    tape: str
    state: str
    steps: int
    halted: bool
    ones: int
    trace: list = field(default_factory=list)


class TuringMachine:
    def __init__(self, rules: dict, start: str, halt_states=("halt",), blank: str = "_"):
        for (_, _), (_, move, _) in rules.items():
            if move not in "LRS":
                raise ValueError(f"bad move {move!r}")
        self.rules, self.start, self.halt_states, self.blank = rules, start, set(halt_states), blank

    def run(self, tape: str = "", max_steps: int = 100_000, trace: bool = False) -> RunResult:
        cells = {i: c for i, c in enumerate(tape) if c != self.blank}
        pos, state, steps, log = 0, self.start, 0, []
        while state not in self.halt_states and steps < max_steps:
            sym = cells.get(pos, self.blank)
            rule = self.rules.get((state, sym))
            if rule is None:
                break  # no applicable rule: the machine is stuck
            write, move, new_state = rule
            if trace:
                log.append((steps, state, pos, sym))
            if write == self.blank:
                cells.pop(pos, None)
            else:
                cells[pos] = write
            pos += {"L": -1, "R": 1, "S": 0}[move]
            state, steps = new_state, steps + 1
        if cells:
            lo, hi = min(cells), max(cells)
            out = "".join(cells.get(i, self.blank) for i in range(lo, hi + 1))
        else:
            out = ""
        halted = state in self.halt_states
        return RunResult(out, state, steps, halted, sum(c == "1" for c in cells.values()), log)


def binary_increment() -> TuringMachine:
    """Adds one to a binary number written most significant bit first."""
    return TuringMachine({
        ("scan", "0"): ("0", "R", "scan"),
        ("scan", "1"): ("1", "R", "scan"),
        ("scan", "_"): ("_", "L", "carry"),
        ("carry", "1"): ("0", "L", "carry"),
        ("carry", "0"): ("1", "S", "halt"),
        ("carry", "_"): ("1", "S", "halt"),
    }, start="scan")


def unary_add() -> TuringMachine:
    """'111+11' -> '11111'."""
    return TuringMachine({
        ("q0", "1"): ("1", "R", "q0"),
        ("q0", "+"): ("1", "R", "q1"),
        ("q1", "1"): ("1", "R", "q1"),
        ("q1", "_"): ("_", "L", "q2"),
        ("q2", "1"): ("_", "S", "halt"),
    }, start="q0")


def palindrome() -> TuringMachine:
    """Accepts palindromes over {a, b} by erasing matching ends. Ends in 'accept' or 'reject'."""
    rules = {("q0", "_"): ("_", "S", "accept")}
    for ch, other, scan, check in (("a", "b", "A", "A2"), ("b", "a", "B", "B2")):
        rules[("q0", ch)] = ("_", "R", scan)
        rules[(scan, "a")] = ("a", "R", scan)
        rules[(scan, "b")] = ("b", "R", scan)
        rules[(scan, "_")] = ("_", "L", check)
        rules[(check, ch)] = ("_", "L", "back")
        rules[(check, other)] = (other, "S", "reject")
        rules[(check, "_")] = ("_", "S", "accept")
    rules[("back", "a")] = ("a", "L", "back")
    rules[("back", "b")] = ("b", "L", "back")
    rules[("back", "_")] = ("_", "R", "q0")
    return TuringMachine(rules, start="q0", halt_states=("accept", "reject"))


def busy_beaver_3() -> TuringMachine:
    """The 3-state, 2-symbol champion for ones: writes six 1s and halts after 14 steps."""
    return TuringMachine({
        ("A", "0"): ("1", "R", "B"), ("A", "1"): ("1", "R", "H"),
        ("B", "0"): ("0", "R", "C"), ("B", "1"): ("1", "R", "B"),
        ("C", "0"): ("1", "L", "C"), ("C", "1"): ("1", "L", "A"),
    }, start="A", halt_states=("H",), blank="0")
