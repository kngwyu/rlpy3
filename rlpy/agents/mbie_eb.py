"""Simple version of MBIE-EB
Paper:An analysis of model-based Interval Estimation for Markov
Decision Processes (Strehl and Littman, 2008)
Link: https://doi.org/10.1016/j.jcss.2007.08.009
"""
import numpy as np
from rlpy.representations import Enumerable
from .agent import Agent
from ._vi_impl import compute_q_values

__author__ = "Yuji Kanagawa"


class MBIE_EB(Agent):
    """
    Simplified version of MBIE-EB algorithm,
    which executes VI only when the episode ends.
    """

    def __init__(
        self,
        *args,
        beta=0.1,
        seed=1,
        spread_prior=False,
        show_reward=False,
        vi_threshold=1e-6,
    ):
        """
        :param beta: β parameter in MBIB-EB
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

        self.sas_count = np.zeros((n_states, n_actions, n_states))

        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_cap = self.representation.domain.episode_cap
        self.update_steps = 0
        self.show_reward = show_reward
        self.vi_threshold = vi_threshold

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
            for a in range(self.n_actions):
                n = self.sa_count[s, a]
                if n == 0:
                    continue
                r = self.r_sum[s, a] / n
                r_sample[s, a] = r + self.beta / np.sqrt(n)
                p_sample[s, a] = self.sas_count[s, a] / n
        if show_reward and hasattr(self.representation.domain, "show_reward"):
            self.representation.domain.show_reward(r_sample.mean(axis=-1))
        return r_sample, p_sample

    def _solve_sampled_mdp(self):
        r, p = self._sample_mdp(show_reward=self.show_reward)
        q_value, _ = compute_q_values(
            r, p, self.ep_cap, self.discount_factor, self.vi_threshold
        )

        self.representation.weight_vec = q_value.T.flatten()
        self.update_steps += 1

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self._update_prior(s, a, r, terminal, ns)
        if terminal is False:
            return
        self._solve_sampled_mdp()
