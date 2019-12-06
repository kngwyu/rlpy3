import copy
import logging
import numpy as np
from rlpy.agents import SARSA, Q_Learning
from rlpy.agents import GreedyGQ
from rlpy.policies import eGreedy
from rlpy.representations import Representation


class MockRepresentation(Representation):
    def __init__(self):
        """
        :param domain: the problem :py:class:`~rlpy.domains.domain.Domain` to learn
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        """
        self.expected_step_cached = {}
        self.state_space_dims = 1
        self.actions_num = 1
        self.discretization = 3
        self.features_num = 4
        self.weight = np.zeros((self.actions_num, self.features_num))
        self._phi_sa_cache = np.empty((self.actions_num, self.features_num))
        self.logger = logging.getLogger(self.__class__.__name__)

    def phi_non_terminal(self, s):
        ret = np.zeros(self.features_num)
        ret[s[0]] = 1.0
        return ret

    def feature_type(self):
        return float


def test_deepcopy():
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = SARSA(pol, rep, 0.9, lambda_=0.0)
    copied_agent = copy.deepcopy(agent)
    assert agent.lambda_ == copied_agent.lambda_


def test_sarsa_valfun_chain():
    """
        Check if SARSA computes the value function of a simple Markov chain correctly.
        This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = SARSA(pol, rep, 0.9, lambda_=0.0)
    for i in range(1000):
        if i % 4 == 3:
            continue
        agent.learn(
            np.array([i % 4]),
            [0],
            0,
            1.0,
            np.array([(i + 1) % 4]),
            [0],
            0,
            (i + 2) % 4 == 0,
        )
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_sarsalambda_valfun_chain():
    """
    Check if SARSA(λ) computes the value function of a simple Markov chain correctly.
    This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = SARSA(pol, rep, 0.9, lambda_=0.5)
    for i in range(1000):
        if i % 4 == 3:
            agent.episode_terminated()
            continue
        agent.learn(
            np.array([i % 4]),
            [0],
            0,
            1.0,
            np.array([(i + 1) % 4]),
            [0],
            0,
            (i + 2) % 4 == 0,
        )
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_qlearn_valfun_chain():
    """
    Check if Q-Learning computes the value function of a simple Markov chain correctly.
    This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = Q_Learning(pol, rep, 0.9, lambda_=0.0)
    for i in range(1000):
        if i % 4 == 3:
            continue
        agent.learn(
            np.array([i % 4]),
            [0],
            0,
            1.0,
            np.array([(i + 1) % 4]),
            [0],
            0,
            (i + 2) % 4 == 0,
        )
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_qlambda_valfun_chain():
    """
    Check if Q(λ) computes the value function of a simple Markov chain correctly.
    This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = Q_Learning(pol, rep, 0.9, lambda_=0.5)
    for i in range(1000):
        if i % 4 == 3:
            agent.episode_terminated()
            continue
        agent.learn(
            np.array([i % 4]),
            [0],
            0,
            1.0,
            np.array([(i + 1) % 4]),
            [0],
            0,
            (i + 2) % 4 == 0,
        )
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)


def test_ggq_valfun_chain():
    """
    Check if Greedy-GQ computes the value function of a simple Markov chain correctly.
    This only tests value function estimation, only one action possible
    """
    rep = MockRepresentation()
    pol = eGreedy(rep)
    agent = GreedyGQ(pol, rep, lambda_=0.0, discount_factor=0.9)
    for i in range(1000):
        if i % 4 == 3:
            agent.episode_terminated()
            continue
        agent.learn(
            np.array([i % 4]),
            [0],
            0,
            1.0,
            np.array([(i + 1) % 4]),
            [0],
            0,
            (i + 2) % 4 == 0,
        )
    V_true = np.array([2.71, 1.9, 1, 0])
    np.testing.assert_allclose(rep.weight_vec, V_true)
