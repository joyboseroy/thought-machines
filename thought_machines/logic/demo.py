from .engine import KnowledgeBase
from .syllogism import check_argument, valid_forms


def main():
    print("== Syllogisms ==")
    v = check_argument("All humans are mortal.", "Socrates is a human.", "Socrates is mortal.")
    print(v.summary(), v.terms)
    v = check_argument("All mammals are animals.", "All mammals are warm-blooded.",
                       "Some warm-blooded are animals.")
    print(v.summary())
    v = check_argument("All cats are animals.", "All dogs are animals.", "All dogs are cats.")
    print(v.summary())
    print(f"valid forms, modern reading:      {len(valid_forms(False))} of 256")
    print(f"valid forms, traditional reading: {len(valid_forms(True))} of 256")

    print("\n== Calculus ratiocinator ==")
    kb = KnowledgeBase()
    kb.tell("human(?x) -> mortal(?x)")
    kb.tell("parent(?a, ?b), parent(?b, ?c) -> grandparent(?a, ?c)")
    kb.tell("human(socrates)")
    kb.tell("parent(sophroniscus, socrates)")
    kb.tell("parent(socrates, lamprocles)")
    print("passes to fixpoint:", kb.run())
    print(kb.explain("mortal(socrates)"))
    print(kb.explain("grandparent(sophroniscus, lamprocles)"))
    print("who is a grandparent?", kb.ask("grandparent(?g, ?c)"))


if __name__ == "__main__":
    main()
