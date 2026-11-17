"""Run every learner on every task under the same conditions and tabulate the results.

    python -m thought_machines.benchmark.runner            quick settings
    python -m thought_machines.benchmark.runner --full     more sizes and repetitions
    python -m thought_machines.benchmark.runner --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time

from .learners import LEARNERS, TRIBES, make
from .tasks import DESCRIPTIONS, TASKS, make_split

QUICK = dict(sizes=(16, 64), noises=(0.0, 0.1), reps=3)
FULL = dict(sizes=(8, 16, 32, 64, 128), noises=(0.0, 0.1, 0.2), reps=8)


def run(tasks=None, learners=None, sizes=(16, 64), noises=(0.0,), reps=3, seed=0):
    tasks = list(tasks or TASKS)
    learners = list(learners or LEARNERS)
    records = []
    for task in tasks:
        for n in sizes:
            for noise in noises:
                for rep in range(reps):
                    # every learner sees the identical split
                    rng = random.Random(f"{seed}/{task}/{n}/{noise}/{rep}")
                    Xtr, ytr, Xte, yte = make_split(task, n, noise, rng)
                    for name in learners:
                        t0 = time.perf_counter()
                        model = make(name, seed=rep).fit(Xtr, ytr)
                        fit_s = time.perf_counter() - t0
                        preds = model.predict_all(Xte)
                        acc = sum(p == t for p, t in zip(preds, yte)) / len(yte)
                        records.append(dict(task=task, learner=name, tribe=TRIBES[name], n_train=n,
                                            noise=noise, rep=rep, accuracy=acc, fit_seconds=fit_s,
                                            size=model.size()))
    return records


def mean_acc(records, **where):
    vals = [r["accuracy"] for r in records if all(r[k] == v for k, v in where.items())]
    return statistics.mean(vals) if vals else float("nan")


def format_table(records, noise: float = 0.0) -> str:
    tasks = sorted({r["task"] for r in records}, key=list(TASKS).index)
    sizes = sorted({r["n_train"] for r in records})
    learners = sorted({r["learner"] for r in records}, key=list(LEARNERS).index)
    lines = []
    for task in tasks:
        lines.append(f"\n{task}: {DESCRIPTIONS[task]}   (label noise {noise:.0%}, test accuracy on unseen inputs)")
        lines.append(f"{'learner':<12} {'tribe':<14}" + "".join(f"{'n=' + str(n):>8}" for n in sizes))
        for name in learners:
            row = "".join(f"{mean_acc(records, task=task, learner=name, n_train=n, noise=noise):>8.2f}"
                          for n in sizes)
            lines.append(f"{name:<12} {TRIBES[name]:<14}{row}")
    return "\n".join(lines)


def average_over_tasks(records, noise: float = 0.0) -> dict:
    sizes = sorted({r["n_train"] for r in records})
    return {name: {n: statistics.mean(mean_acc(records, task=t, learner=name, n_train=n, noise=noise)
                                      for t in TASKS if any(r["task"] == t for r in records))
                   for n in sizes}
            for name in LEARNERS if any(r["learner"] == name for r in records)}


def mean_fit_seconds(records) -> dict:
    return {name: statistics.mean(r["fit_seconds"] for r in records if r["learner"] == name)
            for name in LEARNERS if any(r["learner"] == name for r in records)}


def write_csv(records, path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def report(records) -> str:
    out = []
    for noise in sorted({r["noise"] for r in records}):
        out.append(f"\n##### label noise {noise:.0%}")
        out.append(format_table(records, noise))
        avg = average_over_tasks(records, noise)
        sizes = sorted({r["n_train"] for r in records})
        out.append(f"\n{'mean over tasks':<27}" + "".join(f"{'n=' + str(n):>8}" for n in sizes))
        for name, by_n in avg.items():
            out.append(f"{name:<12} {TRIBES[name]:<14}" + "".join(f"{by_n[n]:>8.2f}" for n in sizes))
    secs = mean_fit_seconds(records)
    out.append("\nmean fit time (s): " + ", ".join(f"{k} {v:.3f}" for k, v in secs.items()))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    cfg = FULL if args.full else QUICK
    records = run(**cfg)
    print(report(records))
    if args.csv:
        write_csv(records, args.csv)
        print(f"\nwrote {len(records)} rows to {args.csv}")


if __name__ == "__main__":
    main()
