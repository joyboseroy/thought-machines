from .runner import QUICK, report, run


def main():
    print("One Task, Five Minds (plus one hybrid): same tasks, same data splits, fixed hyperparameters.")
    print("Quick settings. Run `python -m thought_machines.benchmark.runner --full` for the larger version.")
    print(report(run(**QUICK)))


if __name__ == "__main__":
    main()
