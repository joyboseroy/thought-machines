from .bounded import best_row, compare


def main():
    print("Choosing the best of 1000 options. Net utility = value - cost x looks.")
    for cost in (0.0, 0.001, 0.01, 0.1):
        rows = compare(cost=cost, trials=1000)
        ideal = next(r for r in rows if r.name.startswith("ideal"))
        best = best_row(rows)
        sat = rows[-1]
        print(f"\ncost per look = {cost}")
        print(f"  {'agent':<18} {'looks':>7} {'value':>7} {'net':>8} {'regret':>7}")
        shown = []
        for r in (ideal, best, sat):
            if r not in shown:
                shown.append(r)
        for r in shown:
            tag = "  <- best" if r is best else ""
            print(f"  {r.name:<18} {r.looks:>7.1f} {r.value:>7.2f} {r.net:>8.2f} {r.regret:>7.2f}{tag}")


if __name__ == "__main__":
    main()
