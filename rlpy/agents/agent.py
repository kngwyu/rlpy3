"""Standard Control Agent. """
from abc import ABC, abstractmethod
import numpy as np
import logging

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


class Agent(ABC):

    """Learning Agent for obtaining good policices.

    The Agent receives observations from the Domain and incorporates their
    new information into the representation, policy, etc. as needed.

    In a typical Experiment, the Agent interacts with the Domain in discrete
    timesteps.
    At each Experiment timestep the Agent receives some observations from the Domain
    which it uses to update the value function Representation of the Domain
    (ie, on each call to its :py:meth:`~rlpy.agents.agent.Agent.learn` function).
    The Policy is used to select an action to perform.
    This process (observe, update, act) repeats until some goal or fail state,
    determined by the Domain, is reached. At this point the
    :py:class:`~rlpy.experiments.experiment.Experiment` determines
    whether the agent starts over or has its current policy tested
    (without any exploration).

    :py:class:`~rlpy.agents.agent.Agent` is a base class that provides the basic
    framework for all RL agents. It provides the methods and attributes that
    allow child classes to interact with the
    :py:class:`~rlpy.domains.domain.Domain`,
    :py:class:`~rlpy.representations.Representation`,
    :py:class:`~rlpy.policies.Policy.Policy`, and
    :py:class:`~rlpy.experiments.experiment.Experiment` classes within the
    RLPy library.

    .. note::
        All new agent implementations should inherit from this class.

    """

    def __init__(self, policy, representation, discount_factor, seed=1):
        """initialization.
        :param policy: the :py:class:`~rlpy.policies.Policy.Policy` to use
            when selecting actions.
        :param representation: the :py:class:`~rlpy.representations.Representation`
            to use in learning the value function.
        :param discount_factor: the discount factor of the optimal policy which
            should be  learned
        """
        self.representation = representation
        self.policy = policy
        self.discount_factor = discount_factor
        self.logger = logging.getLogger("rlpy.agents." + self.__class__.__name__)
        # a new stream of random numbers for each agent
        self.random_state = np.random.RandomState(seed=seed)
        #: number of seen episodes
        self.episode_count = 0
        #: The eligibility trace, which marks states as eligible for a learning
        #: update. Used by \ref agents.SARSA.SARSA "SARSA" agent when the
        #: parameter lambda is set. See:
        #: http://www.incompleteideas.net/sutton/book/7/node1.html
        self.eligibility_trace = None

    def set_seed(self, seed):
        """
        Set random seed
        """
        self.random_state.seed(seed)
        self.representation.set_seed(seed=seed)
        self.policy.set_seed(seed=seed)

    @abstractmethod
    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        """
        This function receives observations of a single transition and
        learns from it.

        .. note::
            Each inheriting class (Agent) must implement this method.

        :param s: original state
        :param p_actions: possible actions in the original state
        :param a: action taken
        :param r: obtained reward
        :param ns: next state
        :param np_actions: possible actions in the next state
        :param na: action taken in the next state
        :param terminal: boolean indicating whether next state (ns) is terminal
        """
        pass

    def episode_terminated(self):
        """
        This function adjusts all necessary elements of the agent at the end of
        the episodes.

        .. note::
            Every agent must call this function at the end of the learning if the
            transition led to terminal state.

        """
        # Increase the number of episodes
        self.episode_count += 1
        self.representation.episode_terminated()
        # Set eligibility Traces to zero if it is end of the episode
        if self.eligibility_trace is not None:
            self.eligibility_trace = np.zeros_like(self.eligibility_trace)


class DescentAlgorithm:
    """
    Abstract base class that contains step-size control methods for (stochastic)
    descent algorithms such as TD Learning, Greedy-GQ etc.
    """

    # Valid selections for the ``learn_rate_decay_mode``.
    VALID_DECAY_MODES = ["dabney", "boyan", "const", "boyan_const"]

    def __init__(
        self, initial_learn_rate=0.1, learn_rate_decay_mode="dabney", boyan_N0=1000
    ):
        """
        :param initial_learn_rate: Initial learning rate to use (where applicable).

        .. warning::
            ``initial_learn_rate`` should be set to 1 for automatic learning rate;
            otherwise, initial_learn_rate will act as a permanent upper-bound on learn_rate.

        :param learn_rate_decay_mode: The learning rate decay mode (where applicable)
        :param boyan_N0: Initial Boyan rate parameter (when learn_rate_decay_mode='boyan')

        """
        if learn_rate_decay_mode not in self.VALID_DECAY_MODES:
            raise ValueError("Invalid decay mode: {}".format(learn_rate_decay_mode))
        self.initial_learn_rate = initial_learn_rate
        self.learn_rate = initial_learn_rate
        self.learn_rate_decay_mode = learn_rate_decay_mode.lower()
        self.boyan_N0 = boyan_N0
        # Note that initial_learn_rate should be set to 1 for automatic learning rate; otherwise,
        # initial_learn_rate will act as a permanent upper-bound on learn_rate.
        if self.learn_rate_decay_mode == "dabney":
            self.initial_learn_rate = 1.0
            self.learn_rate = 1.0

    def updateLearnRate(
        self, phi, phi_prime, eligibility_trace, discount_factor, nnz, terminal
    ):
        """Computes a new learning rate (learn_rate) for the agent based on
        ``self.learn_rate_decay_mode``.

        :param phi: The feature vector evaluated at state (s) and action (a)
        :param phi_prime_: The feature vector evaluated at the new state (ns) = (s') and action (na)
        :param eligibility_trace: Eligibility trace
        :param discount_factor: The discount factor for learning (gamma)
        :param nnz: The number of nonzero features
        :param terminal: Boolean that determines if the step is terminal or not

        """

        if self.learn_rate_decay_mode == "dabney":
            # We only update learn_rate if this step is non-terminal; else phi_prime becomes
            # zero and the dot product below becomes very large, creating a very
            # small learn_rate
            if not terminal:
                # Automatic learning rate: [Dabney W. 2012]
                # http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
                candid_learn_rate = np.dot(
                    discount_factor * phi_prime - phi, eligibility_trace
                )
                if candid_learn_rate < 0:
                    self.learn_rate = np.minimum(
                        self.learn_rate, -1.0 / candid_learn_rate
                    )
        elif self.learn_rate_decay_mode == "boyan":
            self.learn_rate = (
                self.initial_learn_rate
                * (self.boyan_N0 + 1.0)
                / (self.boyan_N0 + (self.episode_count + 1) ** 1.1)
            )
            # divide by l1 of the features; note that this method is only called if phi != 0
            self.learn_rate /= np.sum(np.abs(phi))
        elif self.learn_rate_decay_mode == "boyan_const":
            # New little change from not having +1 for episode count
            self.learn_rate = (
                self.initial_learn_rate
                * (self.boyan_N0 + 1.0)
                / (self.boyan_N0 + (self.episode_count + 1) ** 1.1)
            )
        elif self.learn_rate_decay_mode == "const":
            self.learn_rate = self.initial_learn_rate
        else:
            self.logger.warn("Unrecognized decay mode ")
