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

## Cross-links worth knowing about

- `symbolist/ilp.py` produces rules that `logic/engine.py` then runs. Induction and deduction close a loop.
- `causal/` runs on the exact Bayesian network in `bayes/network.py`. An intervention is graph surgery on it.
- `hybrid/rule_logit.py` is built on the tree from `symbolist/`.
- `benchmark/` uses learners from five other pillars through one interface.
- `active/demo.py` reuses the number-game hypothesis space from `bayes/`.
- `memory/` and `spiking/` are two views of the same question: how a network of simple units holds a pattern.

## Questions each tier hands to the next

- Tier 2 asks which learner wins. The benchmark answers it per task, and the answer changes with the task.
- Tier 4 asks what a learner should do when it can choose its data or act on the world.
- Tier 6 asks whether any of the abstractions above can be built from units that only emit spikes.
