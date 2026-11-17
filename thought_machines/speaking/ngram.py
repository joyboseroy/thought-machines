"""Speaking, part 1: a statistical model of language, and what it buys a listener.

Bennett's fifth breakthrough, in humans, is speaking: a shared code that moves learned
knowledge between minds. The oldest formal model of such a code is the n-gram: the
probability of the next word depends on the previous n - 1 words, with counts smoothed
so that unseen combinations are unlikely but not impossible.

Two things are shown:

* Context helps, and so does smoothing. Held-out perplexity falls from unigram to
  bigram to trigram on text from a small grammar, and an unsmoothed model is
  infinitely surprised by any unseen word pair.
* A listener who has a language model can repair a damaged message. This is the noisy
  channel: choose the word that best fits its neighbours. Speaking is not only
  transmission but error-correcting transmission.
"""

from __future__ import annotations

import math
import random

BOS, EOS = "<s>", "</s>"


class ToyGrammar:
    """A small English-like grammar with agreement, so context carries real information."""

    SUBJECTS = [("the dog", "s"), ("the cat", "s"), ("a bird", "s"), ("the dogs", "p"), ("the cats", "p")]
    VERBS = {"s": ["sees", "chases", "likes"], "p": ["see", "chase", "like"]}
    OBJECTS = ["the fish", "a ball", "the mouse", "a toy"]
    PLACES = ["in the garden", "on the mat", "near the house"]

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def sentence(self):
        subject, number = self.rng.choice(self.SUBJECTS)
        words = subject.split() + [self.rng.choice(self.VERBS[number])] + self.rng.choice(self.OBJECTS).split()
        if self.rng.random() < 0.5:
            words += self.rng.choice(self.PLACES).split()
        return words

    def corpus(self, n: int):
        return [self.sentence() for _ in range(n)]


class NGramModel:
    """Order-n model with add-k smoothing. `k = 0` is the unsmoothed maximum likelihood model."""

    def __init__(self, n: int = 2, k: float = 0.1):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n, self.k = n, k
        self.counts, self.context_counts, self.vocab = {}, {}, {EOS}

    def _pad(self, sentence):
        return [BOS] * (self.n - 1) + list(sentence) + [EOS]

    def fit(self, sentences):
        for s in sentences:
            self.vocab.update(s)
            padded = self._pad(s)
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - self.n + 1:i])
                self.counts[(ctx, padded[i])] = self.counts.get((ctx, padded[i]), 0) + 1
                self.context_counts[ctx] = self.context_counts.get(ctx, 0) + 1
        return self

    def prob(self, context, word):
        ctx = tuple(([BOS] * (self.n - 1) + list(context))[-(self.n - 1):]) if self.n > 1 else ()
        num = self.counts.get((ctx, word), 0) + self.k
        den = self.context_counts.get(ctx, 0) + self.k * len(self.vocab)
        return num / den if den > 0 else 1.0 / len(self.vocab)

    def log_prob(self, sentence):
        padded = self._pad(sentence)
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            p = self.prob(padded[:i][-(self.n - 1):] if self.n > 1 else [], padded[i])
            total += math.log(p) if p > 0 else float("-inf")
        return total

    def perplexity(self, sentences):
        logp = sum(self.log_prob(s) for s in sentences)
        tokens = sum(len(s) + 1 for s in sentences)
        return math.exp(-logp / tokens) if logp > float("-inf") else float("inf")

    def generate(self, seed: int = 0, max_len: int = 20):
        rng, out = random.Random(seed), []
        while len(out) < max_len:
            words = sorted(self.vocab)
            weights = [self.prob(out, w) for w in words]
            w = rng.choices(words, weights=weights)[0]
            if w == EOS:
                break
            out.append(w)
        return out


def corrupt(sentence, position: int, replacement: str):
    out = list(sentence)
    out[position] = replacement
    return out


def noisy_channel_repair(model: NGramModel, damaged, position: int):
    """Replace the word at `position` with the one that makes the sentence most probable."""
    best, best_lp = damaged[position], float("-inf")
    for w in sorted(model.vocab - {EOS}):
        lp = model.log_prob(corrupt(damaged, position, w))
        if lp > best_lp:
            best, best_lp = w, lp
    return corrupt(damaged, position, best)
