"""Wrapper agent for count based bonus
"""
from collections import defaultdict
from rlpy.representations import Hashable
from .agent import Agent


class CountBasedBonus(Agent):
    """
    A meta agent which wraps a agent and give him count-based bonuses.
    """

    COUNT_MODES = ["s", "s-a", "s-a-ns"]

    def __init__(self, agent, count_mode="s-a", beta=0.05):
        self.agent = agent
        if not isinstance(agent.representation, Hashable):
            return ValueError("CountBasedBonus requires hashable represenation!!!")
        if count_mode not in self.COUNT_MODES:
            return ValueError("Count mode {} is not supported!".format(count_mode))
        self.counter = defaultdict(int)
        self.count_mode = self.COUNT_MODES.index(count_mode)
        self.beta = beta

    @property
    def policy(self):
        return self.agent.policy

    @property
    def representation(self):
        return self.agent.representation

    def set_seed(self, seed):
        self.agent.set_seed(seed)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        key = self._make_key(s, a, ns)
        self.counter[key] += 1
        bonus = self.beta * self.counter[key] ** -0.5
        self.agent.learn(s, p_actions, a, r + bonus, ns, np_actions, na, terminal)

    def episode_terminated(self):
        self.agent.episode_terminated()

    def _make_key(self, s, a, ns):
        s_hash = self.agent.representation.state_hash(s)
        if self.count_mode == 0:
            return s_hash
        elif self.count_mode == 1:
            return s_hash, a
        else:
            ns_hash = self.agent.representation.state_hash(ns)
            return s_hash, a, ns_hash
