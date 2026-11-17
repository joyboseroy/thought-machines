import random
import statistics as st

from .ngram import NGramModel, ToyGrammar, corrupt, noisy_channel_repair
from .ratchet import cumulative_culture


def main():
    train = ToyGrammar(0).corpus(300)
    test = ToyGrammar(1).corpus(200)

    print("== A language model: how surprised is it by new sentences? ==")
    print("grammar with subject-verb agreement (the dogs see, the dog sees); lower perplexity is better")
    for n, name in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
        print(f"  {name:<8} perplexity {NGramModel(n, 0.1).fit(train).perplexity(test):>6.2f}")
    small = train[:30]
    print("with only 30 training sentences and a trigram model:")
    print(f"  no smoothing   perplexity {NGramModel(3, 0.0).fit(small).perplexity(test)}")
    print(f"  add-0.1        perplexity {NGramModel(3, 0.1).fit(small).perplexity(test):.2f}")
    print("A single unseen word pair makes the unsmoothed model infinitely surprised.")
    gen = NGramModel(3, 0.01).fit(train)
    print("a sentence it generates:", " ".join(gen.generate(seed=3)))

    print("\n== A listener with a language model repairs damaged messages ==")
    print("one word in each of 100 sentences is replaced by a wrong word; can the listener restore it?")
    vocab = sorted(gen.vocab - {"</s>"})
    for n, name in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
        model, rng, ok = NGramModel(n, 0.01).fit(train), random.Random(0), 0
        for s in test[:100]:
            pos = rng.randrange(len(s))
            wrong = rng.choice([w for w in vocab if w != s[pos]])
            ok += noisy_channel_repair(model, corrupt(s, pos, wrong), pos) == s
        print(f"  {name:<8} restores {ok}% of them")
    print("More context means better repair. Agreement is only visible to a model with context.")

    print("\n== The ratchet: language lets one generation hand over what it learned ==")
    print("fitness (fraction of 100 bits correct), mean of 10 runs; each lifetime has 60 trials")
    gens = (1, 2, 5, 10, 30)
    print(f"{'':<34}" + "".join(f"{'gen ' + str(g):>8}" for g in gens))
    for label, social, err in (("no language (start afresh)", False, 0.0), ("language, perfect channel", True, 0.0),
                               ("language, 5% errors", True, 0.05), ("language, 20% errors", True, 0.2),
                               ("language, 40% errors", True, 0.4)):
        runs = [cumulative_culture(30, 100, 60, err, social, s) for s in range(10)]
        print(f"{label:<34}" + "".join(f"{st.mean(r[g - 1] for r in runs):>8.2f}" for g in gens))
    print("Without a channel, every generation re-learns the same ground. With one, knowledge")
    print("accumulates until transmission errors cancel what a single lifetime can add.")


if __name__ == "__main__":
    main()
