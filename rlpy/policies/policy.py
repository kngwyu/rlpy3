"""Policy base class"""
from rlpy.tools import className, discrete_sample
import numpy as np
import logging
from abc import ABC, abstractmethod

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


class Policy(ABC):

    """The Policy determines the discrete action that an
    :py:class:`~rlpy.agents.agent.Agent` will take  given its
    :py:class:`~rlpy.representations.Representation`.

    The Agent learns about the :py:class:`~rlpy.domains.domain.Domain`
    as the two interact.
    At each step, the Agent passes information about its current state
    to the Policy; the Policy uses this to decide what discrete action the
    Agent should perform next (see :py:meth:`~rlpy.policies.Policy.Policy.pi`) \n

    The Policy class is a base class that provides the basic framework for all
    policies. It provides the methods and attributes that allow child classes
    to interact with the Agent and Representation within the RLPy library. \n

    .. note::
        All new policy implementations should inherit from Policy.

    """

    DEBUG = False

    def __init__(self, representation, seed=1):
        """
        :param representation: the :py:class:`~rlpy.representations.Representation`
            to use in learning the value function.
        """

        self.representation = representation
        # An object to record the print outs in a file
        self.logger = logging.getLogger("rlpy.policies." + self.__class__.__name__)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState(seed=seed)

    def set_seed(self, seed):
        """
        Set random seed
        """
        self.random_state.seed(seed)

    @abstractmethod
    def pi(self, s, terminal, p_actions):
        """
        *Abstract Method:*\n Select an action given a state.

        :param s: The current state
        :param terminal: boolean, whether or not the *s* is a terminal state.
        :param p_actions: a list / array of all possible actions in *s*.
        """
        pass

    def turnOffExploration(self):
        """
        *Abstract Method:* \n Turn off exploration (e.g., epsilon=0 in epsilon-greedy)
        """
        pass

    # [turnOffExploration code]

    # \b ABSTRACT \b METHOD: Turn exploration on. See code
    # \ref Policy_turnOnExploration "Here".
    # [turnOnExploration code]
    def turnOnExploration(self):
        """
        *Abstract Method:* \n
        If :py:meth:`~rlpy.policies.Policy.Policy.turnOffExploration` was called
        previously, reverse its effects (e.g. restore epsilon to its previous,
        possibly nonzero, value).
        """
        pass

    def printAll(self):
        """ Prints all class information to console. """
        print(className(self))
        print("=======================================")
        for property, value in vars(self).items():
            print(property, ": ", value)


class DifferentiablePolicy(Policy):
    def pi(self, s, terminal, p_actions):
        """Sample action from policy"""
        p = self.probabilities(s, terminal)
        return discrete_sample(p)

    @abstractmethod
    def dlogpi(self, s, a):
        """derivative of the log probabilities of the policy"""
        pass

    def prob(self, s, a):
        """
        probability of chosing action a given the state s
        """
        v = self.probabilities(s, False)
        return v[a]

    @property
    def theta(self):
        return self.representation.weight_vec

    @theta.setter
    def theta(self, v):
        self.representation.weight_vec = v

    @abstractmethod
    def probabilities(self, s, terminal):
        """
        returns a vector of num_actions length containing the normalized
        probabilities for taking each action given the state s
        """
        pass
