import unittest

from thought_machines.logic.engine import KnowledgeBase
from thought_machines.logic.syllogism import NAMES, check_argument, valid_forms


class SyllogismTests(unittest.TestCase):
    def test_counts_match_the_classical_result(self):
        self.assertEqual(len(valid_forms(False)), 15)
        self.assertEqual(len(valid_forms(True)), 24)

    def test_valid_forms_are_exactly_the_named_ones(self):
        self.assertEqual(set(valid_forms(True)), set(NAMES))

    def test_darapti_needs_existential_import(self):
        self.assertNotIn(("AAI", 3), valid_forms(False))
        self.assertIn(("AAI", 3), valid_forms(True))

    def test_barbara_from_english(self):
        v = check_argument("All humans are mortal.", "Socrates is a human.", "Socrates is mortal.")
        self.assertEqual((v.mood, v.figure, v.name), ("AAA", 1, "Barbara"))
        self.assertTrue(v.valid_modern)

    def test_undistributed_middle_is_invalid(self):
        v = check_argument("All cats are animals.", "All dogs are animals.", "All dogs are cats.")
        self.assertEqual((v.mood, v.figure), ("AAA", 2))
        self.assertFalse(v.valid_modern or v.valid_traditional)

    def test_premise_order_does_not_matter(self):
        a = check_argument("All humans are mortal.", "Socrates is a human.", "Socrates is mortal.")
        b = check_argument("Socrates is a human.", "All humans are mortal.", "Socrates is mortal.")
        self.assertEqual((a.mood, a.figure), (b.mood, b.figure))


class EngineTests(unittest.TestCase):
    def test_socrates(self):
        kb = KnowledgeBase().tell("human(?x) -> mortal(?x)").tell("human(socrates)")
        kb.run()
        self.assertEqual(kb.ask("mortal(socrates)"), [{}])
        self.assertEqual(kb.ask("mortal(plato)"), [])

    def test_two_step_chain_and_join(self):
        kb = KnowledgeBase()
        kb.tell("parent(?a, ?b), parent(?b, ?c) -> grandparent(?a, ?c)")
        kb.tell("parent(a, b)").tell("parent(b, c)").tell("parent(b, d)")
        kb.run()
        self.assertEqual(kb.ask("grandparent(a, ?x)"), [{"?x": "c"}, {"?x": "d"}])

    def test_recursion_reaches_fixpoint(self):
        kb = KnowledgeBase()
        kb.tell("edge(?a, ?b) -> path(?a, ?b)")
        kb.tell("edge(?a, ?b), path(?b, ?c) -> path(?a, ?c)")
        for a, b in (("1", "2"), ("2", "3"), ("3", "4"), ("4", "2")):
            kb.tell(f"edge({a}, {b})")
        kb.run()
        self.assertEqual(len(kb.ask("path(1, ?x)")), 3)

    def test_explanation_reaches_given_facts(self):
        kb = KnowledgeBase().tell("human(?x) -> mortal(?x)").tell("human(socrates)")
        kb.run()
        text = kb.explain("mortal(socrates)")
        self.assertIn("[by human(?x) -> mortal(?x)]", text)
        self.assertIn("human(socrates)  [given]", text)

    def test_unsafe_rule_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeBase().tell("human(?x) -> loves(?x, ?y)")


if __name__ == "__main__":
    unittest.main()
