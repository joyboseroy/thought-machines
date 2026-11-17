"""Aristotle's categorical syllogisms, checked by exhaustive model enumeration.

There are three terms: S (minor), P (major) and M (middle). Each of the three
terms splits the world into 2**3 = 8 regions (in/out of S, P, M). A *model* is a
choice of which regions are inhabited, so there are 2**8 = 256 models.

A statement is true or false in a model:

    A  All X are Y        no inhabited region has X and not Y
    E  No X are Y         no inhabited region has X and Y
    I  Some X are Y       some inhabited region has X and Y
    O  Some X are not Y   some inhabited region has X and not Y

A syllogism is valid when every model making both premises true also makes the
conclusion true. Under the modern (Boolean) reading, terms may be empty and 15
forms survive. Under the traditional reading, every term is assumed to have
members ("existential import") and 9 more forms become valid, 24 in all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

TERMS = ("S", "P", "M")
REGIONS = list(product((False, True), repeat=3))  # (in S, in P, in M)

# Each figure is (major premise terms, minor premise terms). Conclusion is always S-P.
FIGURES = {
    1: (("M", "P"), ("S", "M")),
    2: (("P", "M"), ("S", "M")),
    3: (("M", "P"), ("M", "S")),
    4: (("P", "M"), ("M", "S")),
}

NAMES = {
    # valid on both readings
    ("AAA", 1): "Barbara", ("EAE", 1): "Celarent", ("AII", 1): "Darii", ("EIO", 1): "Ferio",
    ("EAE", 2): "Cesare", ("AEE", 2): "Camestres", ("EIO", 2): "Festino", ("AOO", 2): "Baroco",
    ("AII", 3): "Datisi", ("IAI", 3): "Disamis", ("OAO", 3): "Bocardo", ("EIO", 3): "Ferison",
    ("AEE", 4): "Camenes", ("IAI", 4): "Dimaris", ("EIO", 4): "Fresison",
    # valid only if the terms are assumed non-empty
    ("AAI", 1): "Barbari", ("EAO", 1): "Celaront", ("EAO", 2): "Cesaro", ("AEO", 2): "Camestros",
    ("AAI", 3): "Darapti", ("EAO", 3): "Felapton", ("AAI", 4): "Bamalip",
    ("AEO", 4): "Calemos", ("EAO", 4): "Fesapo",
}


@lru_cache(maxsize=None)
def all_models(existential_import: bool = False):
    models = []
    for mask in range(1 << len(REGIONS)):
        model = frozenset(r for i, r in enumerate(REGIONS) if mask >> i & 1)
        if existential_import and not all(any(r[t] for r in model) for t in range(3)):
            continue
        models.append(model)
    return tuple(models)


def _in(region, term: str) -> bool:
    return region[TERMS.index(term)]


def holds(form: str, x: str, y: str, model) -> bool:
    """Is the statement `form` about terms x, y true in this model?"""
    if form == "A":
        return all((not _in(r, x)) or _in(r, y) for r in model)
    if form == "E":
        return all(not (_in(r, x) and _in(r, y)) for r in model)
    if form == "I":
        return any(_in(r, x) and _in(r, y) for r in model)
    if form == "O":
        return any(_in(r, x) and not _in(r, y) for r in model)
    raise ValueError(f"unknown statement form {form!r}")


def counterexample(mood: str, figure: int, existential_import: bool = False):
    """Return a model where the premises hold and the conclusion fails, or None."""
    (a1, b1), (a2, b2) = FIGURES[figure]
    for model in all_models(existential_import):
        if (holds(mood[0], a1, b1, model) and holds(mood[1], a2, b2, model)
                and not holds(mood[2], "S", "P", model)):
            return model
    return None


def is_valid(mood: str, figure: int, existential_import: bool = False) -> bool:
    return counterexample(mood, figure, existential_import) is None


def all_forms():
    for fig in (1, 2, 3, 4):
        for mood in product("AEIO", repeat=3):
            yield "".join(mood), fig


def valid_forms(existential_import: bool = False):
    return [f for f in all_forms() if is_valid(*f, existential_import)]


# ---------------------------------------------------------------- natural language

_PATTERNS = [
    ("O", re.compile(r"^some (.+?) (?:are|is) not (?:an? )?(.+)$")),
    ("I", re.compile(r"^some (.+?) (?:are|is an?) (.+)$")),
    ("E", re.compile(r"^no (.+?) (?:are|is an?) (.+)$")),
    ("A", re.compile(r"^all (.+?) (?:are|is an?) (.+)$")),
    ("A", re.compile(r"^(.+?) (?:is|are) (?:an? )?(.+)$")),  # singular: "Socrates is a man"
]


def _norm(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r"^(an?|the) ", "", term)
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        term = term[:-1]  # crude singularisation, only needs to be consistent
    return term


def parse_statement(text: str):
    """'All humans are mortal.' -> ('A', 'human', 'mortal')."""
    s = text.strip().rstrip(".").lower()
    for form, pat in _PATTERNS:
        m = pat.match(s)
        if m:
            return form, _norm(m.group(1)), _norm(m.group(2))
    raise ValueError(f"cannot parse statement: {text!r}")


@dataclass
class Verdict:
    mood: str
    figure: int
    name: str | None
    terms: dict
    valid_modern: bool
    valid_traditional: bool

    def summary(self) -> str:
        label = self.name or "unnamed form"
        if self.valid_modern:
            status = "valid"
        elif self.valid_traditional:
            status = "valid only if every term has members (existential import)"
        else:
            status = "invalid"
        return f"{self.mood}-{self.figure} ({label}): {status}"


def check_argument(premise1: str, premise2: str, conclusion: str) -> Verdict:
    """Classify and check a two-premise categorical argument written in English."""
    c_form, c_s, c_p = parse_statement(conclusion)
    stmts = [parse_statement(premise1), parse_statement(premise2)]
    major = next((s for s in stmts if c_p in s[1:]), None)
    minor = next((s for s in stmts if c_s in s[1:] and s is not major), None)
    if major is None or minor is None:
        raise ValueError("premises must contain the conclusion's predicate and subject")
    shared = ({major[1], major[2]} & {minor[1], minor[2]}) - {c_s, c_p}
    if len(shared) != 1:
        raise ValueError("premises must share exactly one middle term")
    m = shared.pop()
    major_m_subject = major[1] == m
    minor_m_subject = minor[1] == m
    figure = {(True, False): 1, (False, False): 2, (True, True): 3, (False, True): 4}[
        (major_m_subject, minor_m_subject)
    ]
    mood = major[0] + minor[0] + c_form
    return Verdict(
        mood=mood, figure=figure, name=NAMES.get((mood, figure)),
        terms={"S": c_s, "P": c_p, "M": m},
        valid_modern=is_valid(mood, figure, False),
        valid_traditional=is_valid(mood, figure, True),
    )
