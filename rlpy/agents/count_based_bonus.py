"""Wrapper agent for count based bonus
"""
from collections import defaultdict
import numpy as np
from rlpy.representations import Enumerable, Hashable
from .agent import Agent


class CountBasedBonus(Agent):
    """
    A meta agent which wraps a agent and give him count-based bonuses.
    """

    COUNT_MODES = ["s", "s-a", "s-a-ns"]

    def __init__(self, agent, count_mode="s-a", beta=0.05, show_reward=False):
        self.agent = agent
        if not isinstance(agent.representation, Hashable):
            return ValueError("CountBasedBonus requires hashable represenation!!!")
        if count_mode not in self.COUNT_MODES:
            return ValueError("Count mode {} is not supported!".format(count_mode))
        self.count_mode = self.COUNT_MODES.index(count_mode)
        if isinstance(agent.representation, Enumerable):
            self.counter = np.zeros(self._counter_shape())
            self._bonus_cache = np.ones(self.counter.shape[0]) * beta
        elif show_reward:
            return ValueError(
                "You cannot use show_reward for not-enumerable representations!"
            )
        else:
            self.counter = defaultdict(int)
        self.beta = beta
        self.show_reward = show_reward

    @property
    def policy(self):
        return self.agent.policy

    @property
    def representation(self):
        return self.agent.representation

    def _counter_shape(self):
        n_states = self.representation.features_num
        n_actions = self.representation.domain.actions_num
        if self.count_mode == 0:
            return n_states
        elif self.count_mode == 1:
            return n_states, n_actions
        else:
            return n_states, n_actions, n_states

    def set_seed(self, seed):
        self.agent.set_seed(seed)

    def _update_bonus_cache(self, s):
        s_id = self.representation.state_id(s)
        count_mean = self.counter[s_id].mean()
        self._bonus_cache[s_id] = self.beta * count_mean ** -0.5

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        key = self._make_key(s, a, ns)
        self.counter[key] += 1
        count = self.counter[key]
        if self.show_reward:
            self._update_bonus_cache(s)
        bonus = self.beta * count ** -0.5
        self.agent.learn(s, p_actions, a, r + bonus, ns, np_actions, na, terminal)
        if (
            terminal
            and self.show_reward
            and hasattr(self.representation.domain, "show_reward")
        ):
            self.representation.domain.show_reward(self._bonus_cache)

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
