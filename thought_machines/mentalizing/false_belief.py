"""Mentalizing, part 2: other people can be wrong, and you can predict how.

The Sally-Anne task. Sally puts a marble in the basket and leaves. While she is away,
Anne moves it to the box. Where will Sally look when she comes back?

A predictor with no model of minds says: where the marble really is, the box. A
predictor that tracks what each person has witnessed says: the basket, because Sally
never saw the move. The second answer is what children over about four give, and what
the first-order belief tracker below computes. A control scene where Sally watches the
move shows that both predictors agree whenever beliefs are true.

The tracker is also nested one level: what Anne thinks Sally believes is computed from
what Anne herself saw of Sally's presence.
"""

from __future__ import annotations


class Scene:
    """A log of events, each remembering who was present to see it."""

    def __init__(self, agents):
        self.agents = list(agents)
        self.present = set(agents)
        self.events = []  # (witnesses, object, location)

    def leave(self, agent):
        self.present.discard(agent)

    def enter(self, agent):
        self.present.add(agent)

    def place(self, obj, location):
        """Put or move an object. Only those present see it."""
        self.events.append((frozenset(self.present), obj, location))

    def reality(self, obj):
        return next(loc for _, o, loc in reversed(self.events) if o == obj)

    def belief(self, agent, obj):
        """Where `agent` thinks the object is: the last placement they witnessed."""
        for witnesses, o, loc in reversed(self.events):
            if o == obj and agent in witnesses:
                return loc
        return None

    def belief_about_belief(self, observer, other, obj):
        """Where `observer` thinks `other` thinks the object is.

        The observer only knows who was present at the events the observer saw.
        """
        for witnesses, o, loc in reversed(self.events):
            if o == obj and observer in witnesses:
                if other in witnesses:
                    return loc
        return None

    def predict_search(self, agent, obj, mentalizing: bool):
        """Where will `agent` look? A mentalizer uses their belief, the other uses the world."""
        return self.belief(agent, obj) if mentalizing else self.reality(obj)


def sally_anne(sally_watches_the_move: bool = False) -> Scene:
    scene = Scene(["Sally", "Anne"])
    scene.place("marble", "basket")
    if not sally_watches_the_move:
        scene.leave("Sally")
    scene.place("marble", "box")  # Anne moves it
    scene.enter("Sally")
    return scene
