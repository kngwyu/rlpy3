"""Posterior Sampling for Reinforcement Learning
Paper: https://arxiv.org/abs/1306.0940, https://arxiv.org/abs/1607.00215
Based on the author's code: https://github.com/iosband/TabulaRL
"""
import numpy as np
from rlpy.representations import Enumerable
from .agent import Agent
from ._vi_impl import compute_q_values, compute_q_values_opt

__author__ = "Yuji Kanagawa"


class PSRL(Agent):
    """Posterior Sampling for Reinforcement Learning
    """

    def __init__(
        self,
        *args,
        alpha0=1.0,
        mu0=0.0,
        tau0=1.0,
        tau=1.0,
        seed=1,
        spread_prior=False,
        show_reward=False,
    ):
        """
        :param alpha0: Prior weight for uniform Dirichlet.
        :param mu0: Prior mean rewards.
        :param tau0: Precision of prior mean rewards.
        :param tau: Precision of reward noise.
        :param spread_prior: Use alpha0/n_states as alpha0
        """
        super().__init__(*args, seed=seed)
        if not isinstance(self.representation, Enumerable):
            raise ValueError("PSRL works only with a tabular representation.")

        self.epsilon = 0.0
        self.tau = tau
        n_states = self.representation.features_num
        n_actions = self.representation.domain.actions_num

        self.r_prior_mu = np.ones((n_states, n_actions)) * mu0
        self.r_prior_tau = np.ones((n_states, n_actions)) * tau0

        if spread_prior:
            alpha0 /= n_states
        self.p_prior = (
            np.ones((n_states, n_actions, n_states), dtype=np.float32) * alpha0
        )
        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_cap = self.representation.domain.episode_cap
        self.update_steps = 0
        self.show_reward = show_reward

    def _update_prior(self, s, a, reward, terminal, ns):
        s_id = self.representation.state_id(s)
        tau_old = self.r_prior_tau[s_id, a]
        tau_new = tau_old + self.tau
        self.r_prior_tau[s_id, a] = tau_new
        mu_old = self.r_prior_mu[s_id, a]
        self.r_prior_mu[s_id, a] = (mu_old * tau_old + reward * self.tau) / tau_new
        if not terminal:
            ns_id = self.representation.state_id(ns)
            self.p_prior[s_id, a, ns_id] += 1

    def _sample_mdp(self, show_reward=False):
        r_sample = np.zeros_like(self.r_prior_mu)
        p_sample = np.zeros_like(self.p_prior)
        for s in range(self.n_states):
            mu, tau = self.r_prior_mu[s], self.r_prior_tau[s]
            r_sample[s] = mu + self.random_state.randn(self.n_actions) / np.sqrt(tau)
            for a in range(self.n_actions):
                p_sample[s, a] = self.random_state.dirichlet(self.p_prior[s, a])
        if show_reward and hasattr(self.representation.domain, "show_reward"):
            self.representation.domain.show_reward(r_sample.mean(axis=-1))
        return r_sample, p_sample

    def _solve_sampled_mdp(self):
        r, p = self._sample_mdp(show_reward=self.show_reward)
        q_value, _ = compute_q_values(r, p, self.ep_cap, self.discount_factor)

        self.representation.weight_vec = q_value.T.flatten()
        self.update_steps += 1

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self._update_prior(s, a, r, terminal, ns)
        if terminal is False:
            return
        self._solve_sampled_mdp()


class OptimisticPSRL(PSRL):
    def __init__(self, *args, n_samples=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def _solve_optimistic_mdp(self):
        r, p = self._sample_mdp(show_reward=self.show_reward)
        q_max, v_max = compute_q_values(r, p, self.ep_cap, self.discount_factor)

        for i in range(1, self.n_samples):
            r, p = self._sample_mdp()
            q, v = compute_q_values(r, p, self.ep_cap, self.discount_factor)
            v_max = np.maximum(v_max, v)
            q_max = np.maximum(q_max, q)

        self.representation.weight_vec = q_max.T.flatten()
        self.update_steps += 1

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self._update_prior(s, a, r, terminal, ns)
        if terminal is False:
            return
        self._solve_optimistic_mdp()


class GaussianPSRL(PSRL):
    def __init__(self, *args, scaling=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling

    def _gen_bonus(self):
        r_bonus = self.random_state.randn(self.r_prior_mu.shape)
        r_bonus = self.scaling * r_bonus / np.sqrt(self.r_prior_tau)
        p_bonus = self.random_state.randn(r_bonus.shape)
        p_bonus = self.scaling * p_bonus / np.sqrt(self.p_prior.sum(axis=-1))
        return r_bonus, p_bonus

    def _solve_sampled_mdp(self):
        r, p = self._sample_mdp(show_reward=False)
        r_bonus, p_bonus = self._gen_bonus()
        if self.show_reward and hasattr(self.representation.domain, "show_reward"):
            self.representation.domain.show_reward((r + r_bonus).mean(axis=-1))

        q_value, _ = compute_q_values(
            r, p, r_bonus, p_bonus, self.ep_cap, self.discount_factor
        )

        self.representation.weight_vec = q_value.T.flatten()
        self.update_steps += 1

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self._update_prior(s, a, r, terminal, ns)
        if terminal is False:
            return
        self._solve_sampled_mdp()


class UCBVI(PSRL):
    def _gen_bonus(self, h=1.0):
        r_bonus = self.scaling * np.ones(self.r_prior_mu.shape)
        r_bonus = r_bonus * np.sqrt(2.0 * np.log(2.0 + h) / self.r_prior_tau)
        p_bonus = self.scaling * np.ones(self.r_prior_mu.shape)
        p_bonus = p_bonus * np.sqrt(2.0 * np.log(2.0 + h) / self.p_prior.sum(axis=-1))
        return r_bonus, p_bonus
