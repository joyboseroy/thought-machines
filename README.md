# thought-machines

An executable atlas of mathematical theories of mind, from Aristotle's syllogisms to spiking
neurons. Plain Python, no dependencies, 99 tests, and a full demo run in about 15 seconds.

```bash
git clone https://github.com/joyboseroy/thought-machines
cd thought-machines
python -m thought_machines list           # the tiers
python -m thought_machines demo all       # run everything (about 15 seconds)
python -m thought_machines demo spiking   # or one pillar
python -m unittest discover -s tests -t . # 99 tests
```

Python 3.11 or later (tested on 3.11, 3.12 and 3.13). No numpy, no PyTorch, no API keys.

---

## Why this exists

In 2026 Tom Griffiths published *The Laws of Thought: The Quest for a Mathematical Theory of
the Mind* (Henry Holt, ISBN 978-1-250-35835-6; William Collins in the UK, ISBN
978-0-00-871148-1). Its subject is the long attempt, starting about three hundred years ago,
to describe thinking with mathematics, from early logic through to the ideas behind modern AI
systems, and the ways those systems still differ from human minds. If you have read it, or any
book like it, the next question for a programmer is the obvious one: can I run these ideas?

A book can explain what a syllogism, a perceptron or Bayes' rule is. It cannot let you change
the training set and watch what happens. This repository is the bench that goes with the
reading. Each idea is a small program you can run, break and measure, with a test that pins
down the result the literature says you should get.

A second book gave the middle of the project its shape. Pedro Domingos, *The Master Algorithm*
(Basic Books, 2015), divides machine learning into five tribes (symbolists, connectionists,
evolutionaries, Bayesians, analogizers) and argues that each has a piece of a general learner.
Whether the tribes really are complementary is an empirical question, so tier 2 implements all
five and tier 3 runs them on the same tasks with the same splits and the same fixed
hyperparameters.

Past the two books, the atlas continues upward into topics that most "algorithms from scratch"
collections skip: causal intervention, choosing experiments, thinking under a cost,
perception as prediction, associative memory, and finally spiking neurons, which are the
substrate that every higher tier is an abstraction of.

This repository is original code. It reproduces no text from either book, and it does not claim
to implement "the" theory of mind or "the" master algorithm. The aim is narrower: make
competing theories executable, measurable and comparable.

## How it relates to *The Laws of Thought*

Publisher material and reviews describe the book as following three mathematical frameworks
for thought: logic, artificial neural networks and probability. This repository's modules
cover the same ground, plus further topics. The table is my own mapping for finding code, not
the book's chapter structure, which this repository does not follow.

| framework in the book | where it lives here | what you can run |
|---|---|---|
| Logic and rules | `logic/`, `computation/`, `symbolist/` | A syllogism checker (15 valid forms, or 24 with existential import), a Horn-clause engine with proof traces, NAND circuits and Turing machines, decision trees and inductive logic |
| Neural networks | `neural/`, `memory/`, `spiking/` | Perceptron and its XOR limit, hand-written backpropagation, Hopfield and sparse distributed memory, spiking neurons with STDP |
| Probability | `bayes/`, `causal/`, `active/`, `predictive/` | Bayes' rule and the number game, Bayesian networks and MCMC, the do-operator, expected information gain, predictive coding |

The title phrase also belongs to George Boole, whose 1854 book *An Investigation of the Laws
of Thought* turned logic into algebra. Tier 0 implements that step: Boolean algebra as gates,
gates as adders, and a machine that computes.

The remaining modules (`search/`, `evolutionary/`, `analogizer/`, `hybrid/`, `benchmark/`,
`rationality/`) follow the Domingos and AI-textbook lineage rather than the book. See
`docs/MAP.md` for how everything connects and `docs/READING.md` for the sources behind each
tier.

## Who gets something out of cloning it

**If you are learning the field.** Each pillar is a few short files, about 3,200 lines of
source in all, readable in one sitting per pillar. Nothing is hidden behind a library call, so when the
demo says a Hopfield network recalls 3 patterns but not 30, you can open `memory/hopfield.py`
and see exactly why.

**If you teach it.** Every demo has a known answer and a test behind it: 15 valid syllogisms
under the modern reading and 24 with existential import, busy beaver 3, the textbook
burglary alarm posterior of 0.284, the LIF firing rate against its analytic formula. Seeds are
fixed, failures are shown (the demo reports how many random starts solve XOR), and toy data is
labelled as toy. You can hand a student a module and a question.

**If you want to compare learners.** `benchmark/` puts a decision tree, an MLP, genetic
programming, naive Bayes, k-nearest neighbours and a hybrid rule learner behind one interface
(`fit`, `predict_all`, `size`) and scores them on identical tasks and splits with fixed
hyperparameters. Adding your own learner means writing one small adapter class. The result is
a quick way to see which family of method suits which kind of problem, and why.

**If you want a sanity check on an intuition.** "Seeing is not doing", "good questions beat
random ones", "memory has a capacity cliff" each have a small experiment you can rerun and
modify in seconds.

It is not a library for production, not tuned for speed, and not a state-of-the-art anything.
The benchmark tasks are small by design.

## What is and is not new

None of the algorithms here is new, and none of the individual implementations is better than
the mature ones. The choice and arrangement are the contribution.

Two existing projects are the closest neighbours.
[aima-python](https://github.com/aimacode/aima-python) implements the algorithms from Russell
and Norvig's textbook, organised by textbook chapter.
[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) provides NumPy
implementations of machine-learning models from linear regression to deep learning, organised
by model type. Both are excellent and you should use them if your goal is to learn the
textbook or the standard model zoo.

What this repository does differently:

1. **The unit is a theory of mind, not a textbook chapter or a model type.** The order of the
   tiers follows the history of ideas, from Aristotle to spiking neurons, so each tier hands a
   question to the next (`docs/MAP.md`).
2. **The tribes are compared, not just implemented.** Five learners from five traditions run
   through one interface on shared tasks, with fixed hyperparameters and the raw rows
   exportable to CSV.
3. **The upper tiers cover material that general ML repositories usually omit:** interventions
   and the backdoor adjustment, expected-information-gain experiment design, bounded
   rationality, predictive coding, sparse distributed memory and spike-timing plasticity.
4. **Zero dependencies.** The whole thing runs on the standard library, so every mechanism can
   be read top to bottom.
5. **Every claim in a demo has a test**, including known textbook numbers.

The experiments in this repository are small illustrations, not research findings. If you want
a research-grade comparison of memory mechanisms under a controlled protocol, that is a
different project: [memory-under-constraints](https://github.com/joyboseroy/memory-under-constraints).

---

## The tiers

| Tier | Pillar | Idea | Mode of thought |
|---|---|---|---|
| 0 | `logic` | Syllogisms and a Horn-clause engine with proof traces | deduction |
| 0 | `computation` | NAND-only gates, adders, Turing machines | mechanical computation |
| 1 | `search` | BFS, DFS, uniform cost, greedy, A* on puzzles | problem solving |
| 2 | `symbolist` | ID3 decision trees; FOIL-style inverse deduction | learning rules |
| 2 | `neural` | Perceptron, hand-written backprop, concept geometry | learning representations |
| 2 | `evolutionary` | Genetic algorithm; genetic programming | evolving structure |
| 2 | `bayes` | Bayes' rule, number game, Bayes nets, MCMC, naive Bayes | updating belief |
| 2 | `analogizer` | k-nearest neighbours; kernel SVM | reasoning by similarity |
| 3 | `hybrid` | RuleLogit: tree structure with probabilistic rule weights | synthesis |
| 3 | `benchmark` | Every tribe on the same tasks and splits | comparison |
| 4 | `causal` | do-operator, backdoor adjustment on Bayes nets | seeing versus doing |
| 4 | `active` | Expected information gain picks the next experiment | machine science |
| 4 | `rationality` | Ideal versus bounded agents when thinking costs | bounded optimality |
| 4 | `predictive` | Perception as minimising prediction error | predictive coding |
| 5 | `memory` | Hopfield networks; Kanerva's sparse distributed memory | associative memory |
| 6 | `spiking` | LIF and Izhikevich neurons, STDP, rank-order coding | the biological floor |

Tier 2 is Domingos's five tribes (symbolists, connectionists, evolutionaries, Bayesians, analogizers). Tiers 4 to 6 go past them: causal reasoning and experimentation (Pearl and the scientific-method tradition), the predictive-mind literature, memory, and finally the spike-level substrate that all the higher tiers are abstractions of. See `docs/MAP.md` for the picture and `docs/READING.md` for the reading list.

## What the experiments show

These come from the demos and tests in this repository, on small problems. They are illustrations, not general claims.

**The tribes disagree in predictable ways.** On hidden boolean rules over 8 features, with fixed untuned hyperparameters (quick settings, 3 repetitions, accuracy on unseen inputs averaged over five tasks):

| learner | tribe | clean, n=16 | clean, n=64 | 10% label noise, n=64 |
|---|---|---|---|---|
| tree | symbolist | 0.84 | 0.92 | 0.80 |
| mlp | connectionist | 0.76 | 0.99 | 0.80 |
| gp | evolutionary | 0.78 | 0.86 | 0.84 |
| naive_bayes | bayesian | 0.68 | 0.75 | 0.73 |
| knn | analogizer | 0.67 | 0.80 | 0.75 |
| rule_logit | hybrid | 0.83 | 0.92 | 0.81 |

Naive Bayes sits at chance on XOR and parity because its independence assumption cannot see an interaction. The network needs enough examples but then wins on clean data. Genetic programming holds up best under label noise. Run `python -m thought_machines.benchmark.runner --full` for more sizes and repetitions, and `--csv out.csv` to keep the raw rows.

**Structure versus parameters.** Genetic programming recovers `x^2 + 3x + 1` exactly from 21 samples on [-5, 5] on every tested seed, and the discovered formula still gives the right answer at x = 20. A fixed-form learner has to be told the form first.

**Seeing is not doing.** In the ice cream and drowning network, `P(drowning | ice cream sold)` is 0.25 but `P(drowning | do(ice cream sold))` is 0.175, the base rate. With a small real effect of 0.04, the naive observed difference is 0.196. Backdoor adjustment on the confounder recovers the interventional answer exactly.

**Asking good questions.** Choosing yes/no questions by expected information gain finds a hidden value among 64 in 6 questions and starts at the median, which is binary search rediscovered. Random questions have not converged after 40.

**Memory has a cliff.** A 100-unit Hopfield network recalls 3 stored patterns from a 10% corrupted cue essentially always and 30 patterns almost never. Kanerva's memory shows a critical distance beyond which iterated recall drifts away from the stored word.

**Spikes carry timing.** A leaky integrate-and-fire neuron's simulated firing rate matches the analytic formula to within 1.5 Hz. A neuron trained with STDP on inputs that all have the same average rate drives the weights of the correlated group to the maximum (1.0) and leaves the uncorrelated group near 0.5. A rank-order code classifies stimuli under random monotonic distortion at 1.00 where Euclidean template matching gets about 0.89.

## Design rules

1. Core models are written without modern AI frameworks, so each mechanism is readable in one sitting.
2. Every claim in a demo has a test behind it. Known results (15 and 24 valid syllogisms, busy beaver 3, the textbook burglary alarm number 0.284) are checked, not assumed.
3. Randomness is seeded. Failures are reported, not hidden: the demo says how many random starts solve XOR.
4. Toy data is labelled as toy.
5. Hyperparameters are fixed across tasks in the benchmark. Nothing is tuned per learner per task.

## Layout

```
thought_machines/
  logic/ computation/ search/
  symbolist/ neural/ evolutionary/ bayes/ analogizer/
  hybrid/ benchmark/
  causal/ active/ rationality/ predictive/
  memory/ spiking/
  cli.py
tests/
docs/   MAP.md  READING.md  ROADMAP.md
```

## Known limits

- The syllogism parser singularises words crudely, so irregular plurals will confuse it.
- The inference engine handles definite Horn clauses only: no negation, no disjunction.
- The ILP learner is a small covering learner with short rule bodies, not a full ILP system.
- The kernel SVM uses the Pegasos update with no bias term, which suits the symmetric toy problems here.
- RuleLogit is a toy in the direction of Domingos's synthesis. It is not a Markov logic network.
- The benchmark is boolean rules on 8 features. It measures accuracy and fit time but not yet extrapolation, calibration or compositionality.
- The spiking tier is single neurons and a single plastic neuron. It is not a network simulator.

## References

Full per-tier reading is in `docs/READING.md`. The sources the code is most directly modelled
on:

**The two anchor books**

* Griffiths, T. L. (2026). *The Laws of Thought: The Quest for a Mathematical Theory of the
  Mind*. Henry Holt and Company. ISBN 978-1-250-35835-6. (UK edition: William Collins,
  ISBN 978-0-00-871148-1.)
* Domingos, P. (2015). *The Master Algorithm: How the Quest for the Ultimate Learning Machine
  Will Remake Our World*. Basic Books.

**Logic and computation**

* Boole, G. (1854). *An Investigation of the Laws of Thought*. Walton and Maberly.
* Turing, A. M. (1936). On computable numbers, with an application to the
  Entscheidungsproblem. *Proceedings of the London Mathematical Society*.

**Learning rules and representations**

* Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning* 1(1).
* Quinlan, J. R. (1990). Learning logical definitions from relations. *Machine Learning* 5(3).
* Rosenblatt, F. (1958). The perceptron. *Psychological Review* 65(6).
* Minsky, M. and Papert, S. (1969). *Perceptrons*. MIT Press.
* Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986). Learning representations by
  back-propagating errors. *Nature* 323.
* Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of
  Michigan Press.
* Koza, J. R. (1992). *Genetic Programming*. MIT Press.
* Cover, T. and Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions
  on Information Theory* 13(1).
* Cortes, C. and Vapnik, V. (1995). Support-vector networks. *Machine Learning* 20(3).
* Richardson, M. and Domingos, P. (2006). Markov logic networks. *Machine Learning* 62.

**Probability, causation and inference**

* Tenenbaum, J. B. (1999). Bayesian modeling of human concept learning. *Advances in Neural
  Information Processing Systems 11*.
* Metropolis, N. et al. (1953). Equation of state calculations by fast computing machines.
  *Journal of Chemical Physics* 21(6).
* Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
* Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Annals of
  Mathematical Statistics* 27(4).
* Gigerenzer, G. and Goldstein, D. G. (1996). Reasoning the fast and frugal way. *Psychological
  Review* 103(4).
* Rao, R. P. N. and Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature
  Neuroscience* 2(1).

**Memory and spiking neurons**

* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective
  computational abilities. *Proceedings of the National Academy of Sciences* 79(8).
* Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
* Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural
  Networks* 14(6).
* Song, S., Miller, K. D. and Abbott, L. F. (2000). Competitive Hebbian learning through
  spike-timing-dependent synaptic plasticity. *Nature Neuroscience* 3(9).
* Thorpe, S., Delorme, A. and Van Rullen, R. (2001). Spike-based strategies for rapid
  processing. *Neural Networks* 14(6-7).
* Gerstner, W., Kistler, W. M., Naud, R. and Paninski, L. (2014). *Neuronal Dynamics*.
  Cambridge University Press.

Bibliographic details outside the two anchor books are given from memory of well-known works.
Please check them before citing, and open an issue if one is wrong.

## Status, authorship and citation

Version 0.2. Every pillar has a demo and tests, and the roadmap (`docs/ROADMAP.md`) lists what
would make the benchmark scientifically stronger: extrapolation, calibration, compositionality,
transfer and interpretability scores. The experiments are small and have not been
peer reviewed.

Written by Joy Bose. The code was developed with assistance from Claude. Treat it like any
research code: run the tests, read a module before relying on it, and check the numbers
against the references.

If you use it, please cite it using `CITATION.cff`.

## License

MIT.
