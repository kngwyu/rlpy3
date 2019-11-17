"""Domain base class"""
from abc import ABC, abstractmethod
from copy import deepcopy
import logging
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


class Domain(ABC):
    """
    The Domain controls the environment in which the
    :py:class:`~rlpy.agents.agent.Agent` resides as well as the reward function the
    Agent is subject to.

    The Agent interacts with the Domain in discrete timesteps called
    *episodes* (see :py:meth:`~rlpy.domains.domain.Domain.step`).
    At each step, the Agent informs the Domain what indexed action it wants to
    perform.  The Domain then calculates the effects this action has on the
    environment and updates its internal state accordingly.
    It also returns the new state to the agent, along with a reward/penalty,
    and whether or not the episode is over (thus resetting the agent to its
    initial state).

    This process repeats until the Domain determines that the Agent has either
    completed its goal or failed.
    The :py:class:`~rlpy.experiments.experiment.Experiment` controls this cycle.

    Because agents are designed to be agnostic to the Domain that they are
    acting within and the problem they are trying to solve, the Domain needs
    to completely describe everything related to the task. Therefore, the
    Domain must not only define the observations that the Agent receives,
    but also the states it can be in, the actions that it can perform, and the
    relationships between the three.

    The Domain class is a base clase that provides the basic framework for all
    domains. It provides the methods and attributes that allow child classes
    to interact with the Agent and Experiment classes within the RLPy library.
    domains should also provide methods that provide visualization of the
    Domain itself and of the Agent's learning
    (:py:meth:`~rlpy.domains.domain.Domain.show_domain` and
    :py:meth:`~rlpy.domains.domain.Domain.show_learning` respectively) \n
    All new domain implementations should inherit from :py:class:`~rlpy.domains.domain.domain`.

    .. note::
        Though the state *s* can take on almost any value, if a dimension is not
        marked as 'continuous' then it is assumed to be integer.

    """

    def __init__(
        self,
        actions_num,
        statespace_limits,
        discount_factor=0.9,
        continuous_dims=None,
        episode_cap=None,
    ):
        """
        :param actions_num: The number of Actions the agent can perform
        :param discount_factor: The discount factor by which rewards are reduced
        :param statespace_limits: Limits of each dimension of the state space.
        Each row corresponds to one dimension and has two elements [min, max]
        :param state_space_dims: Number of dimensions of the state space
        :param continuous_dims: List of the continuous dimensions of the domain
        :param episode_cap: The cap used to bound each episode (return to state 0 after)
        """
        self.actions_num = actions_num
        self.statespace_limits = statespace_limits
        self.discount_factor = float(discount_factor)
        if continuous_dims is None:
            self.states_num = int(
                np.prod(self.statespace_limits[:, 1] - self.statespace_limits[:, 0])
            )
            self.continuous_dims = []
        else:
            self.states_num = np.inf
            self.continuous_dims = continuous_dims

        self.episode_cap = episode_cap

        self.random_state = np.random.RandomState()

        self.state_space_dims = self.statespace_limits.shape[0]
        # For discrete domains, limits should be extended by half on each side so that
        # the mapping becomes identical with continuous states.
        # The original limits will be saved in self.discrete_statespace_limits.
        self._extendDiscreteDimensions()

        self.logger = logging.getLogger("rlpy.domains." + self.__class__.__name__)
        self.seed = None

        self.performance = False

    def set_seed(self, seed):
        """
        Set random seed
        """
        self.seed = seed
        self.random_state.seed(seed)

    def __str__(self):
        res = """{self.__class__}:
------------
Dimensions: {self.state_space_dims}
|S|:        {self.states_num}
|A|:        {self.actions_num}
Episode Cap:{self.episode_cap}
Gamma:      {self.discount_factor}
""".format(
            self=self
        )
        return res

    def show(self, a=None, representation=None):
        """
        Shows a visualization of the current state of the domain and that of
        learning.

        See :py:meth:`~rlpy.domains.domain.Domain.show_domain()` and
        :py:meth:`~rlpy.domains.domain.Domain.show_learning()`,
        both called by this method.

        .. note::
            Some domains override this function to allow an optional *s*
            parameter to be passed, which overrides the *self.state* internal
            to the domain; however, not all have this capability.

        :param a: The action being performed
        :param representation: The learned value function
            :py:class:`~rlpy.Representation.Representation.Representation`.

        """
        self.saveRandomState()
        self.show_domain(a=a)
        self.show_learning(representation=representation)
        self.loadRandomState()

    def show_domain(self, a=0):
        """
        *Abstract Method:*\n
        Shows a visualization of the current state of the domain.

        :param a: The action being performed.

        """
        pass

    def show_learning(self, representation):
        """
        *Abstract Method:*\n
        Shows a visualization of the current learning,
        usually in the form of a gridded value function and policy.
        It is thus really only possible for 1 or 2-state domains.

        :param representation: the learned value function
            :py:class:`~rlpy.Representation.Representation.Representation`
            to generate the value function / policy plots.

        """
        pass

    @abstractmethod
    def s0(self):
        """
        Begins a new episode and returns the initial observed state of the Domain.
        Sets self.state accordingly.

        :return: A numpy array that defines the initial domain state.

        """
        pass

    def possible_actions(self, s=None):
        """
        The default version returns an enumeration of all actions [0, 1, 2...].
        We suggest overriding this method in your domain, especially if not all
        actions are available from all states.

        :param s: The state to query for possible actions
            (overrides self.state if ``s != None``)

        :return: A numpy array containing every possible action in the domain.

        .. note::

            *These actions must be integers*; internally they may be handled
            using other datatypes.  See :py:meth:`~rlpy.tools.general_tools.vec2id`
            and :py:meth:`~rlpy.tools.general_tools.id2vec` for converting between
            integers and multidimensional quantities.

        """
        return np.arange(self.actions_num)

    # TODO: change 'a' to be 'aID' to make it clearer when we refer to
    # actions vs. integer IDs of actions?  They aren't always interchangeable.
    @abstractmethod
    def step(self, a):
        """
        *Abstract Method:*\n
        Performs the action *a* and updates the Domain
        state accordingly.
        Returns the reward/penalty the agent obtains for
        the state/action pair determined by *Domain.state*  and the parameter
        *a*, the next state into which the agent has transitioned, and a
        boolean determining whether a goal or fail state has been reached.

        .. note::

            domains often specify stochastic internal state transitions, such
            that the result of a (state,action) pair might vary on different
            calls (see also the :py:meth:`~rlpy.domains.domain.Domain.sampleStep`
            method).
            Be sure to look at unique noise parameters of each domain if you
            require deterministic transitions.


        :param a: The action to perform.

        .. warning::

            The action *a* **must** be an integer >= 0, and might better be
            called the "actionID".  See the class description
            :py:class:`~rlpy.domains.domain.Domain` above.

        :return: The tuple (r, ns, t, p_actions) =
            (Reward [value], next observed state, is_terminal [boolean])

        """
        pass

    def saveRandomState(self):
        """
        Stores the state of the the random generator.
        Using loadRandomState this state can be loaded.
        """
        self.random_state_backup = self.random_state.get_state()

    def loadRandomState(self):
        """
        Loads the random state stored in the self.random_state_backup
        """
        self.random_state.set_state(self.random_state_backup)

    def is_terminal(self):
        """
        Returns ``True`` if the current Domain.state is a terminal one, ie,
        one that ends the episode.  This often results from either a failure
        or goal state being achieved.\n
        The default definition does not terminate.

        :return: ``True`` if the state is a terminal state, ``False`` otherwise.

        """
        return False

    def _extendDiscreteDimensions(self):
        """
        Offsets discrete dimensions by 0.5 so that binning works properly.

        .. warning::

            This code is used internally by the Domain base class.
            **It should only be called once**

        """
        # Store the original limits for other types of calculations
        self.discrete_statespace_limits = self.statespace_limits
        self.statespace_limits = self.statespace_limits.astype("float")
        for d in range(self.state_space_dims):
            if d not in self.continuous_dims:
                self.statespace_limits[d, 0] += -0.5
                self.statespace_limits[d, 1] += +0.5

    @property
    def statespace_width(self):
        return self.statespace_limits[:, 1] - self.statespace_limits[:, 0]

    @property
    def discrete_statespace_width(self):
        return (
            self.discrete_statespace_limits[:, 1]
            - self.discrete_statespace_limits[:, 0]
        )

    def sampleStep(self, a, num_samples):
        """
        Sample a set number of next states and rewards from the domain.
        This function is used when state transitions are stochastic;
        deterministic transitions will yield an identical result regardless
        of *num_samples*, since repeatedly sampling a (state,action) pair
        will always yield the same tuple (r,ns,terminal).
        See :py:meth:`~rlpy.domains.domain.Domain.step`.

        :param a: The action to attempt
        :param num_samples: The number of next states and rewards to be sampled.

        :return: A tuple of arrays ( S[], A[] ) where
            *S* is an array of next states,
            *A* is an array of rewards for those states.

        """
        next_states = []
        rewards = []
        s = self.state.copy()
        for i in range(num_samples):
            r, ns, terminal = self.step(a)
            self.state = s.copy()
            next_states.append(ns)
            rewards.append(r)

        return np.array(next_states), np.array(rewards)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in list(self.__dict__.items()):
            if k == "logger":
                continue
            # This block bandles matplotlib transformNode objects,
            # which cannot be coped
            try:
                setattr(result, k, deepcopy(v, memo))
            except Exception:
                if hasattr(v, "frozen"):
                    setattr(result, k, v.frozen())
                else:
                    import warnings

                    warnings.warn("Skip {} when copying".format(k))
        return result
