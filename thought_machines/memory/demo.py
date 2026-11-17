from .hopfield import recall_rate
from .sdm import SDM, popcount


def main():
    print("== Hopfield network, 100 units, 10% of the cue corrupted ==")
    for p in (3, 8, 15, 30):
        print(f"  {p:>2} stored patterns: exact recall {recall_rate(100, p, 0.1, trials=2):.2f}")
    print("Recall degrades as the memory fills (the classical limit is near 0.14 N patterns).")

    print("\n== Sparse distributed memory, 256 bits, 2000 hard locations ==")
    sdm = SDM(seed=1)
    words = [sdm.random_word() for _ in range(40)]
    for w in words:
        sdm.write(w, w)  # auto-associative
    for flips in (10, 30, 50):
        errs = []
        for w in words[:10]:
            out = sdm.recall(sdm.noisy(w, flips))
            errs.append(popcount(out ^ w))
        exact = sum(e == 0 for e in errs)
        print(f"  cue with {flips:>2} flipped bits: {exact}/10 recalled exactly, "
              f"{10 - exact} diverged to junk")
    print("Iterated recall either converges to the stored word or drifts away. There is a critical")
    print("distance between the two, which is the classic SDM result.")


if __name__ == "__main__":
    main()
