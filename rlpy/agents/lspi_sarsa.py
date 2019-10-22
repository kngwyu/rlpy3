"""
[EXPERIMENTAL] Least-Squares Policy Iteration but with SARSA
"""
from .lspi import LSPI
from .td_control_agents import SARSA

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


class LSPI_SARSA(SARSA):
    """This agent uses SARSA for online learning and calls LSPI on sample_window"""

    def __init__(
        self,
        policy,
        representation,
        discount_factor,
        lspi_iterations=5,
        steps_between_lspi=100,
        sample_window=100,
        tol_epsilon=1e-3,
        re_iterations=100,
        initial_learn_rate=0.1,
        lambda_=0,
        learn_rate_decay_mode="dabney",
        boyan_N0=1000,
    ):
        super().__init__(
            policy,
            representation,
            discount_factor,
            lambda_=lambda_,
            initial_learn_rate=initial_learn_rate,
            learn_rate_decay_mode=learn_rate_decay_mode,
            boyan_N0=boyan_N0,
        )
        self.lspi = LSPI(
            policy,
            representation,
            discount_factor,
            sample_window,
            steps_between_lspi,
            lspi_iterations,
            tol_epsilon,
            re_iterations,
        )

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        """Iterative learning method for the agent.
        :param s: The current state features
        :type s: numpy.ndarray
        :param p_actions: The actions available in state s.
        :type p_actions: numpy.ndarray
        :param a: The action taken by the agent in state s.
        :type a: int
        :param r: The reward received by the agent for taking action a in state s.
        :type r: float
        :param ns: The next state features.
        :type ns: numpy.ndarray
        :param np_actions: The actions available in state ns.
        :type np_actions: numpy.ndarray
        :param na: The action taken by the agent in state ns.
        :type ns: int
        :param terminal: Whether or not ns is a terminal state.
        :type terminal: bool
        """
        self.lspi.process(s, a, r, ns, na, terminal)
        if self.lspi.samples_count % self.lspi.steps_between_lspi == 0:
            self.lspi.representationExpansionLSPI()
            if terminal:
                self.episode_terminated()
        else:
            self.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
