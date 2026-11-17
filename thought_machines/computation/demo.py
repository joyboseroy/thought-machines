from . import circuits as c
from .turing import binary_increment, busy_beaver_3, palindrome, unary_add


def main():
    print("== Truth tables, built from NAND only ==")
    for name, fn in (("AND", c.and_), ("OR", c.or_), ("XOR", c.xor_), ("NOR", c.nor_)):
        rows = " ".join(f"{a}{b}->{o}" for (a, b), o in c.truth_table(fn))
        print(f"{name:>3}: {rows}")

    print("\n== Ripple-carry adder ==")
    for x, y in ((5, 3), (13, 14), (15, 15)):
        out = c.from_bits(c.ripple_add(c.to_bits(x, 4), c.to_bits(y, 4)))
        print(f"{x} + {y} = {out}")

    print("\n== Turing machines ==")
    r = binary_increment().run("1011")
    print(f"increment 1011 -> {r.tape} in {r.steps} steps")
    r = unary_add().run("111+11")
    print(f"unary 111+11 -> {r.tape}")
    for word in ("abba", "abab", "aba", ""):
        r = palindrome().run(word)
        print(f"palindrome({word!r}) -> {r.state}")
    r = busy_beaver_3().run("")
    print(f"busy beaver 3: {r.ones} ones, {r.steps} steps, halted={r.halted}")


if __name__ == "__main__":
    main()
