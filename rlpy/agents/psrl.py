"""Posterior Sampling for Reinforcement Learning
"""
import itertools
import numpy as np
from rlpy.policies import eGreedy
from rlpy.representations import Tabular
from .agent import Agent
from ._vi_impl import compute_q_values

__author__ = "Yuji Kanagawa"


class PSRL(Agent):
    """Posterior Sampling for Reinforcement Learning
    """

    # Minimum number of trajectories required for convergence in which the max
    # bellman error was below the threshold
    MIN_CONVERGED_TRAJECTORIES = 5

    def __init__(
        self,
        representation,
        discount_factor,
        alpha0=1.0,
        mu0=0.0,
        tau0=1.0,
        tau=1.0,
        seed=1,
    ):
        """
        :param representation: the :py:class:`~rlpy.representations.Representation`
            to use in learning the value function.
        :param discount_factor: the discount factor of the optimal policy which
            should be  learned
        :param step_size: Step size parameter to adjust the weights.
        :param alpha0: Prior weight for uniform Dirichlet.
        :param mu0: Prior mean rewards.
        :param tau0: Precision of prior mean rewards.
        :param tau: Precision of reward noise.
        """
        super().__init__(
            eGreedy(representation, epsilon=0.0),
            representation,
            discount_factor,
            seed=seed,
        )
        if not isinstance(representation, Tabular):
            raise ValueError("PSRL works only with a tabular representation.")

        self.epsilon = 0.0
        self.tau = tau
        n_states = self.representation.features_num
        n_actions = self.representation.domain.actions_num

        self.alpha = 1.0

        self.r_prior_mu = np.ones((n_states, n_actions)) * mu0
        self.r_prior_tau = np.ones((n_states, n_actions)) * tau0

        self.p_prior = (
            np.ones((n_states, n_actions, n_states), dtype=np.float32) * alpha0
        )
        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_cap = self.representation.domain.episode_cap
        self.discount_factor = discount_factor

    def _update_prior(self, s, a, reward, terminal, ns):
        s_id = self.representation.hash_for_state_count(s)
        tau_old = self.r_prior_tau[s_id, a]
        tau_new = tau_old + self.tau
        self.r_prior_tau[s_id, a] = tau_new
        mu_old = self.r_prior_mu[s_id, a]
        self.r_prior_mu[s_id, a] = (mu_old * tau_old + reward * self.tau) / tau_new
        ns_id = self.representation.hash_for_state_count(ns)
        self.p_prior[s_id, a, ns_id] += 1

    def _sample_mdp(self):
        r_sample = np.zeros_like(self.r_prior_mu)
        p_sample = np.zeros_like(self.p_prior)
        for s, a in itertools.product(range(self.n_states), range(self.n_actions)):
            mu, tau = self.r_prior_mu[s, a], self.r_prior_tau[s, a]
            r_sample[s, a] = mu + self.random_state.normal() / np.sqrt(tau)
            p_sample[s, a] = self.random_state.dirichlet(self.p_prior[s, a])
        return r_sample, p_sample

    def _solve_sampled_mdp(self):
        r, p = self._sample_mdp()
        q_value = compute_q_values(r, p, self.ep_cap, self.discount_factor)
        self.representation.weight_vec = q_value.T.flatten()

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self._update_prior(s, a, r, terminal, ns)
        if terminal is False:
            return
        self._solve_sampled_mdp()
