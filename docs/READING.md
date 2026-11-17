# Reading list

Books and papers behind each tier. The core shelf and the "mathematical machinery" and "competing theories" lists come from the project's planning notes. The per-tier papers are the ones the code is modelled on. Publication details are given from memory, so check them before citing.

## The two anchor books

- Thomas L. Griffiths, *The Laws of Thought: The Quest for a Mathematical Theory of the Mind* (2026). Henry Holt and Company, ISBN 978-1-250-35835-6; UK edition William Collins, ISBN 978-0-00-871148-1. What mathematical principles could explain thinking.
- Pedro Domingos, *The Master Algorithm* (2015). Five tribes of machine learning and the search for a universal learner.

## The core shelf

- Douglas Hofstadter, *Gödel, Escher, Bach* (1979)
- Tim Crane, *The Mechanical Mind*
- Andy Clark, *Mindware*

## The mathematical machinery

- *Bayesian Models of Cognition*
- David MacKay, *Information Theory, Inference, and Learning Algorithms* (2003)
- Mohri, Rostamizadeh and Talwalkar, *Foundations of Machine Learning*
- Shalev-Shwartz and Ben-David, *Understanding Machine Learning* (2014)
- Kearns and Vazirani, *An Introduction to Computational Learning Theory* (1994)

These three are where the idea of a universal learner meets the mathematics of what can and cannot be learned.

## The competing theories

- Peter Gärdenfors, *Conceptual Spaces: The Geometry of Thought* (2000)
- Jakob Hohwy, *The Predictive Mind* (2013)
- Andy Clark, *Surfing Uncertainty* (2016)
- Parr, Pezzulo and Friston, *Active Inference* (2022)
- Judea Pearl and Dana Mackenzie, *The Book of Why* (2018)
- Gerd Gigerenzer, *Rationality for Mortals* (2008)

## Per tier: what the code is modelled on

**Logic and computation.** Aristotle's *Prior Analytics* for the syllogism. Leibniz's calculus ratiocinator as the program of reasoning by calculation. Boole, *The Laws of Thought* (1854). Turing, "On computable numbers" (1936).

**Symbolists.** Quinlan, "Induction of decision trees" (1986). Quinlan, "Learning logical definitions from relations" (1990), the FOIL algorithm. Muggleton and De Raedt, "Inductive logic programming: theory and methods" (1994).

**Connectionists.** Rosenblatt's perceptron (1958). Minsky and Papert, *Perceptrons* (1969), for the XOR limit. Rumelhart, Hinton and Williams, "Learning representations by back-propagating errors" (1986).

**Evolutionaries.** Holland, *Adaptation in Natural and Artificial Systems* (1975). Koza, *Genetic Programming* (1992).

**Bayesians.** Tenenbaum's work on the number game and the size principle (1999). Russell and Norvig, *Artificial Intelligence: A Modern Approach*, for the burglary alarm network. Metropolis et al. (1953) for the sampler.

**Analogizers.** Cover and Hart, "Nearest neighbor pattern classification" (1967). Cortes and Vapnik, "Support-vector networks" (1995). Shalev-Shwartz et al., "Pegasos" (2007), the update used here.

**Synthesis.** Richardson and Domingos, "Markov logic networks" (2006), the architecture Domingos sketches and this repository does not yet implement.

**Causal.** Pearl, *Causality* (2000), for the do-operator and the backdoor criterion.

**Active learning and bounded rationality.** Lindley (1956) on information gain as the value of an experiment. Simon on satisficing. Gigerenzer and Goldstein, "Reasoning the fast and frugal way" (1996).

**Predictive.** Rao and Ballard, "Predictive coding in the visual cortex" (1999). Friston on the free-energy principle.

**Memory.** Hopfield, "Neural networks and physical systems with emergent collective computational abilities" (1982). Kanerva, *Sparse Distributed Memory* (1988).

**Spiking.** Gerstner, Kistler, Naud and Paninski, *Neuronal Dynamics* (2014). Dayan and Abbott, *Theoretical Neuroscience* (2001). Izhikevich, "Simple model of spiking neurons" (2003). Song, Miller and Abbott, "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity" (2000). Thorpe, Delorme and Van Rullen on rank-order coding (2001). Maass, "Networks of spiking neurons: the third generation of neural network models" (1997).

## Tier 7: the evolutionary axis

**Ordering idea.** Bennett, *A Brief History of Intelligence* (2023). Popular science, used only for the sequence: steering, reinforcing, simulating, mentalizing, speaking.

**Steering.** Berg and Brown (1972) on bacterial run and tumble. Braitenberg, *Vehicles* (1984).

**Reinforcing.** Sutton (1988) on temporal differences. Watkins and Dayan (1992) on Q-learning. Sutton and Barto, *Reinforcement Learning: An Introduction* (2018), for the random walk, cliff walking and the Dyna maze. Schultz, Dayan and Montague (1997) on dopamine as prediction error.

**Simulating.** Sutton (1991) on Dyna. Tolman (1948) on cognitive maps and latent learning.

**Mentalizing.** Baker, Saxe and Tenenbaum (2009) on action understanding as inverse planning. Wimmer and Perner (1983) on false belief.

**Speaking.** Shannon (1948) on the noisy channel. Jurafsky and Martin, *Speech and Language Processing*, on n-gram models and smoothing. Tomasello (1999) on the ratchet effect in cumulative culture.
