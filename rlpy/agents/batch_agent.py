"""An abstract class for Batch Learning agents"""
import numpy as np
from .agent import Agent

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class BatchAgent(Agent):
    """An abstract class for batch agents
    """

    def __init__(self, policy, representation, discount_factor, max_window):
        """
        :param representation: the :py:class:`~rlpy.representations.Representation`
            to use in learning the value function.
        :param policy: the :py:class:`~rlpy.policies.Policy.Policy` to use
            when selecting actions.
        :param discount_factor: the discount factor of the optimal policy which
            should be  learned
        :param max_window: Max size of a batch
        """
        super().__init__(policy, representation, discount_factor=discount_factor)
        self.max_window = max_window
        self.samples_count = 0  #: Number of samples gathered so far

        # Take memory for stored values
        self.data_s = np.zeros((max_window, self.representation.state_space_dims))
        self.data_ns = np.zeros((max_window, self.representation.state_space_dims))
        self.data_a = np.zeros((max_window, 1), dtype=np.uint32)
        self.data_na = np.zeros((max_window, 1), dtype=np.uint32)
        self.data_r = np.zeros((max_window, 1))

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        """Iterative learning method for the agent.

        :param ndarray s: The current state features.
        :param ndarray p_actions: The actions available in state s.
        :param int a: The action taken by the agent in state s.
        :param float r: The reward received by the agent for taking action a in state s.
        :param ndarray ns: The next state features.
        :param ndarray np_actions: The actions available in state ns.
        :param int na: The action taken by the agent in state ns.
        :param bool terminal: Whether or not ns is a terminal state.
        """
        self.store_samples(s, a, r, ns, na, terminal)
        if terminal:
            self.episode_terminated()
        if self.samples_count % self.max_window == 0:
            self.batch_learn()

    def batch_learn(self):
        pass

    def store_samples(self, s, a, r, ns, na, terminal):
        """Process one transition instance."""
        # Save samples
        self.data_s[self.samples_count, :] = s
        self.data_a[self.samples_count] = a
        self.data_r[self.samples_count] = r
        self.data_ns[self.samples_count, :] = ns
        self.data_na[self.samples_count] = na

        self.samples_count += 1
