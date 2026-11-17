"""Command line entry point: `python -m thought_machines demo logic`.

Pillars are listed in tiers, from formal foundations up to spiking neurons.
"""

from __future__ import annotations

import argparse
import importlib

# (name, tier, one line description). Order is the order `demo all` runs them.
PILLAR_LIST = [
    ("logic", "0 foundations", "Aristotle and Leibniz: syllogisms and a forward-chaining engine"),
    ("computation", "0 foundations", "Boole and Turing: NAND circuits, adders, Turing machines"),
    ("search", "1 problem solving", "State-space search and the value of heuristics"),
    ("symbolist", "2 the five tribes", "Decision trees and inverse deduction (learn rules)"),
    ("neural", "2 the five tribes", "Perceptron, XOR, a hand-written MLP, concept geometry"),
    ("evolutionary", "2 the five tribes", "Genetic algorithm and genetic programming (evolve structure)"),
    ("bayes", "2 the five tribes", "Bayes, number game, networks, MCMC, naive Bayes"),
    ("analogizer", "2 the five tribes", "k-nearest neighbours and kernel SVM (reason by similarity)"),
    ("hybrid", "3 synthesis", "RuleLogit: symbolic structure with probabilistic weights"),
    ("benchmark", "3 synthesis", "One task, many minds: the tribes compared on shared tasks"),
    ("causal", "4 beyond the tribes", "Seeing versus doing: interventions on Bayesian networks"),
    ("active", "4 beyond the tribes", "Choosing experiments by expected information gain"),
    ("rationality", "4 beyond the tribes", "Ideal versus bounded agents when thinking has a cost"),
    ("predictive", "4 beyond the tribes", "Predictive coding and a toy of active inference"),
    ("memory", "5 memory", "Hopfield networks and Kanerva's sparse distributed memory"),
    ("spiking", "6 spiking neurons", "LIF and Izhikevich neurons, STDP, rank-order coding"),
]

PILLARS = {name: blurb for name, _, blurb in PILLAR_LIST}
TIERS = {name: tier for name, tier, _ in PILLAR_LIST}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="thought-machines", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the pillars by tier")
    demo = sub.add_parser("demo", help="run a pillar demo")
    demo.add_argument("name", choices=[*PILLARS, "all"])
    args = parser.parse_args(argv)

    if args.cmd == "list":
        last = None
        for name, tier, blurb in PILLAR_LIST:
            if tier != last:
                print(f"\ntier {tier}")
                last = tier
            print(f"  {name:<13} {blurb}")
        return 0
    names = list(PILLARS) if args.name == "all" else [args.name]
    for name in names:
        if len(names) > 1:
            print(f"\n{'=' * 12} {name} (tier {TIERS[name]}) {'=' * 12}")
        importlib.import_module(f"thought_machines.{name}.demo").main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
