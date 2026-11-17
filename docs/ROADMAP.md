# Roadmap

## Done in v0.2

All six v0.1 pillars plus the five tribes, the hybrid learner, the benchmark, causal reasoning, active learning, predictive coding, associative memory and spiking neurons. Every pillar has a demo and tests.

## v0.3: closing the gaps the benchmark exposes

The benchmark currently measures accuracy and fit time on boolean rules. The questions that make the comparison scientifically interesting are not yet measured:

- **Extrapolation.** Train on a range, test outside it. Genetic programming already recovers `x^2 + 3x + 1` and extrapolates exactly; a network and a nearest-neighbour learner should not. This needs a numeric task alongside the boolean ones.
- **Uncertainty.** Who knows when it does not know? A Bayesian regressor with a predictive standard deviation that grows away from the data is the natural reference. Score calibration, not just accuracy.
- **Compositionality.** Train on A and B separately, test on A with B.
- **Transfer.** Learn one task, then a related one, and measure the head start.
- **Interpretability.** The tree prints rules and the genetic program prints a formula. Define a size or description-length score so the comparison is not purely verbal.
- **Scaling.** How accuracy and cost change as the number of distractor features grows.

## v0.3: the hybrid Domingos actually describes

RuleLogit is a toy. The synthesis in *The Master Algorithm* uses Markov logic networks for representation, posterior probability for evaluation, and genetic search plus gradient descent for optimisation. A first step: weighted first-order rules over a small domain, exact inference on a tiny ground network, weights learned by gradient descent and rule structure refined by evolution.

## v0.3: more theories

- **Language.** n-gram models with smoothing, then a small neural language model, scored by perplexity on one corpus.
- **Program induction.** Hypotheses as programs in a small language, a prior that penalises length, posterior by enumeration. This generalises the number game from fixed sets to generated ones.
- **Active inference proper.** The predictive tier has a thermostat toy. A discrete-state agent that minimises expected free energy in a small maze would be the real thing.
- **Conceptual spaces.** Categories as convex regions and prototypes as Voronoi cells, over embeddings learned from data instead of hand-built features.
- **Network-level spiking.** A small recurrent spiking network that stores and replays a sequence. This is the natural capstone of the bottom tier.

## Open questions worth testing

- Does the size principle still help when examples come from a sloppy teacher instead of being sampled from the concept?
- At what per-look cost does a satisficer overtake a fixed-budget sampler, and does the threshold matter more than the budget?
- Where is the critical distance in sparse distributed memory as a function of load, and does it match the textbook prediction?
- Which of the syllogism forms valid only under existential import do people, and models, treat as valid?

## Housekeeping

- Optional plotting extra (`matplotlib`) for posterior curves, search frontiers, learning curves and spike rasters.
- A static web playground for the number game, the search visualiser and a spiking neuron.
