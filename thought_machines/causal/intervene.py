"""Causalists: seeing is not doing.

P(Y | X = x) answers "what do I expect to see if I observe X = x". P(Y | do(X = x))
answers "what happens if I force X = x". They differ whenever a common cause
sits behind both. Interventions are implemented by graph surgery on the exact
Bayesian network from `bayes.network`.
"""

from __future__ import annotations

from itertools import product

from ..bayes.network import BayesNet


def confounded_net() -> BayesNet:
    """Hot weather drives both ice cream sales (X) and drownings (Y). X has no effect on Y."""
    net = BayesNet()
    net.add("Hot", (), {(): 0.5})
    net.add("IceCream", ("Hot",), {(True,): 0.8, (False,): 0.2})
    net.add("Drowning", ("Hot",), {(True,): 0.30, (False,): 0.05})
    return net


def confounded_net_with_effect() -> BayesNet:
    """Same confounder, but ice cream really does raise the drowning risk a little."""
    net = BayesNet()
    net.add("Hot", (), {(): 0.5})
    net.add("IceCream", ("Hot",), {(True,): 0.8, (False,): 0.2})
    net.add("Drowning", ("IceCream", "Hot"), {
        (True, True): 0.35, (True, False): 0.08, (False, True): 0.30, (False, False): 0.05})
    return net


def rain_net() -> BayesNet:
    """Rain wets the grass and makes people carry umbrellas. The sprinkler wets grass too."""
    net = BayesNet()
    net.add("Rain", (), {(): 0.3})
    net.add("Sprinkler", ("Rain",), {(True,): 0.05, (False,): 0.4})
    net.add("Umbrella", ("Rain",), {(True,): 0.9, (False,): 0.05})
    net.add("WetGrass", ("Rain", "Sprinkler"), {
        (True, True): 0.99, (True, False): 0.9, (False, True): 0.9, (False, False): 0.01})
    return net


def observe(net: BayesNet, y: str, x: str, value: bool = True) -> float:
    return net.prob({y: True}, {x: value})


def intervene(net: BayesNet, y: str, x: str, value: bool = True) -> float:
    return net.do(x, value).prob({y: True})


def backdoor(net: BayesNet, y: str, x: str, adjust: list, value: bool = True) -> float:
    """P(y | do(x)) = sum_z P(y | x, z) P(z), valid when `adjust` blocks all backdoor paths."""
    total = 0.0
    for zs in product((False, True), repeat=len(adjust)):
        z = dict(zip(adjust, zs))
        pz = net.prob(z)
        if pz > 0:
            total += net.prob({y: True}, {x: value, **z}) * pz
    return total


def average_effect(net: BayesNet, y: str, x: str) -> float:
    """P(y | do(x=1)) - P(y | do(x=0))."""
    return intervene(net, y, x, True) - intervene(net, y, x, False)
