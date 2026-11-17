# Map

How the pillars relate. Read top to bottom: each tier leans on the one above it.

```
ARISTOTLE   syllogisms: valid reasoning has a shape                     logic
    |
LEIBNIZ     reasoning as calculation                                    logic (engine)
    |
BOOLE       logic as algebra, so gates, so circuits                     computation
    |
TURING      what a mechanical process can compute                       computation
    |
SEARCH      finding a solution in a huge space, and what heuristics buy search
    |
    +---------------------------------------------------------------+
    |                       THE FIVE TRIBES                         |
    |                                                               |
    |  SYMBOLISTS      CONNECTIONISTS    EVOLUTIONARIES             |
    |  learn rules     learn weights     evolve structure           |
    |  symbolist/      neural/           evolutionary/              |
    |                                                               |
    |  BAYESIANS                         ANALOGIZERS                |
    |  update belief                     compare to examples        |
    |  bayes/                            analogizer/                |
    +---------------------------------------------------------------+
    |
SYNTHESIS   symbolic structure + probabilistic weights                hybrid
            all tribes, same tasks, same splits                       benchmark
    |
BEYOND THE TRIBES
    |-- CAUSAL        what happens if I intervene?                    causal
    |-- ACTIVE        what should I test next?                        active
    |-- BOUNDED       what is the best use of limited thinking?       rationality
    |-- PREDICTIVE    perception as error minimisation                predictive
    |
MEMORY      retain and retrieve by content                            memory
    |
SPIKES      the substrate: timing, plasticity, order codes            spiking
```

## A second axis: order of evolutionary emergence (tier 7)

A different ordering, not a continuation of the one above. It follows Bennett's five breakthroughs
and says when each capability appeared, not which one is built on which.

```
STEERING      approach what is good, avoid what is bad (valence)      steering
    |
REINFORCING   learn which actions pay off from reward error          reinforcement
    |
SIMULATING    rehearse in an internal model before acting            simulating
    |
MENTALIZING   model what another mind wants and believes             mentalizing
    |
SPEAKING      a shared code that moves what one mind learned         speaking
              into another

SPIKES (tier 6) are the substrate under every row, not a step after them.
```

## Cross-links worth knowing about

- `symbolist/ilp.py` produces rules that `logic/engine.py` then runs. Induction and deduction close a loop.
- `causal/` runs on the exact Bayesian network in `bayes/network.py`. An intervention is graph surgery on it.
- `hybrid/rule_logit.py` is built on the tree from `symbolist/`.
- `benchmark/` uses learners from five other pillars through one interface.
- `active/demo.py` reuses the number-game hypothesis space from `bayes/`.
- `simulating/` plans by calling `search/core.py` on a problem built from the learned model, and builds on the grid worlds in `reinforcement/`.
- `mentalizing/` reuses the grid movement from `reinforcement/` and runs Bayes' rule over goals, as in `bayes/`.
- `speaking/` stands alone on purpose. Its ratchet is the only place in the repository where knowledge moves between agents.
- `memory/` and `spiking/` are two views of the same question: how a network of simple units holds a pattern.

## Questions each tier hands to the next

- Tier 2 asks which learner wins. The benchmark answers it per task, and the answer changes with the task.
- Tier 4 asks what a learner should do when it can choose its data or act on the world.
- Tier 6 asks whether any of the abstractions above can be built from units that only emit spikes.
- Tier 7 asks, for each capability, what it added over the one before it: reward learning over taxis, a model over reward learning, a model of another mind over a model of the world, and a shared code over all of them.
