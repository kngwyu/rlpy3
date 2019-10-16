"""epsilon-Greedy Policy"""
from .policy import Policy
import numpy as np

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


class eGreedy(Policy):
    """ Greedy policy with epsilon-probability for uniformly random exploration.

    From a given state, it selects the action with the highest expected value
    (greedy with respect to value function), but with some probability
    ``epsilon``, takes a random action instead.
    This explicitly balances the exploration/exploitation tradeoff, and
    ensures that in the limit of infinite samples, the agent will
    have explored the entire domain.
    """

    def __init__(
        self,
        representation,
        epsilon=0.1,
        deterministic=False,
        epsilon_decay=0.0,
        epsilon_min=0.0,
        seed=1,
    ):
        """
        :param representation: The representation that the agent use.
        :param epsilon: Probability of selecting a random action instead of greedy.
        :param deterministic: Select an action deterministically among the best actions.
        :param episilon_decay: if > 0, linealy decays episilon.
        :param episilon_min: The minimum value of epsilon when epsilon_decay > 0.
        :param seed: Random seed used for action sampling.
        """
        self.epsilon = epsilon
        self.deterministic = deterministic
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Temporarily stores value of ``epsilon`` when exploration disabled
        self.old_epsilon = None
        super().__init__(representation, seed)

    def pi(self, s, terminal, p_actions):
        coin = self.random_state.rand()
        eps = self.epsilon
        if self.epsilon_decay > 0 and self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if coin < eps:
            return self.random_state.choice(p_actions)
        else:
            b_actions = self.representation.best_actions(s, terminal, p_actions)
            if self.deterministic:
                return b_actions[0]
            else:
                return self.random_state.choice(b_actions)

    def prob(self, s, terminal, p_actions):
        p = np.ones(len(p_actions)) / len(p_actions)
        p *= self.epsilon
        b_actions = self.representation.best_actions(s, terminal, p_actions)
        if self.deterministic:
            p[b_actions[0]] += 1 - self.epsilon
        else:
            p[b_actions] += (1 - self.epsilon) / len(b_actions)
        return p

    def turnOffExploration(self):
        self.old_epsilon = self.epsilon
        self.epsilon = 0

    def turnOnExploration(self):
        self.epsilon = self.old_epsilon
