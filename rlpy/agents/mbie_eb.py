"""MBIE-EB
"""
import numpy as np
from rlpy.representations import Enumerable
from .agent import Agent
from ._vi_impl import compute_q_values

__author__ = "Yuji Kanagawa"


class MBIE_EB(Agent):
    def __init__(
        self, *args, beta=0.1, seed=1, spread_prior=False, show_reward=False,
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

        n_states = self.representation.features_num
        n_actions = self.representation.domain.actions_num

        self.beta = beta

        self.sa_count = np.zeros((n_states, n_actions))
        self.r_sum = np.zeros((n_states, n_actions))

        self.sas_count = np.zeros((n_states, n_actions, n_states)) + 0.5

        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_cap = self.representation.domain.episode_cap
        self.update_steps = 0
        self.show_reward = show_reward

    def _update_prior(self, s, a, reward, terminal, ns):
        s_id = self.representation.state_id(s)
        self.sa_count[s_id, a] += 1
        self.r_sum[s_id, a] += reward
        if not terminal:
            ns_id = self.representation.state_id(ns)
            self.sas_count[s_id, a, ns_id] += 1

    def _sample_mdp(self, show_reward=False):
        r_sample = np.zeros_like(self.sa_count)
        p_sample = np.zeros_like(self.sas_count)
        for s in range(self.n_states):
            n = self.sa_count[s]
            r = self.r_sum[s] / (n + 1.0)
            r_sample[s] = r + self.beta / (np.sqrt(n) + 1.0)
            for a in range(self.n_actions):
                sum_ = self.sas_count[s, a].sum()
                p_sample[s, a] = self.sas_count[s, a] / sum_
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
