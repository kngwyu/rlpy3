"""UCRL2
"""
import numpy as np
from rlpy.representations import Enumerable
from .agent import Agent
from .evi import EVI


def chernoff(it, n, delta, sqrt_c, log_c):
    return np.sqrt(sqrt_c * np.log(log_c * (it + 1) / delta) / np.maximum(1, n))


class UCRL(Agent):
    def __init__(
        self,
        policy,
        representation,
        discount_factor,
        r_max=1.0,
        alpha_r=1.0,
        alpha_p=1.0,
        seed=1,
        show_reward=False,
    ):
        """
        :param r_max: Upper bound of reward
        :param alpha_r: multiplicative factor for the concentration bound on rewards
            (default is r_max)
        :param alpha_p: multiplicative factor for the concentration bound on transition
            probabilities (default is 1)
        """
        super().__init__(policy, representation, discount_factor, seed=seed)
        if not isinstance(self.representation, Enumerable):
            raise ValueError("UCRL2 works only with a enumerable representation.")

        # Parameters
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p

        n_states = self.representation.features_num
        n_actions = self.representation.domain.actions_num

        # Setup solver
        self.opt_solver = EVI(
            n_states,
            self.representation.actions_per_state(),
            "chernoff",
            seed,
            gamma=discount_factor,
        )

        # Statistics
        self.prob = np.ones((n_states, n_actions, n_states)) / n_states
        self.counter = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
        self.visited_sa = set()

        self.estimated_rewards = np.ones((n_states, n_actions)) * (r_max + 99)
        # Only for SMDP
        self.estimated_holding_times = np.ones((n_states, n_actions))

        self.n_obs = np.zeros((n_states, n_actions), dtype=np.int64)
        self.n_k = np.zeros_like(self.n_obs)

        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_cap = self.representation.domain.episode_cap
        self.update_steps = 0
        self.show_reward = show_reward

        self.solver_policy = np.zeros(n_states, dtype=np.int64)
        self.solver_policy_indices = np.zeros_like(self.solver_policy)
        self.iteration = 0
        self.delta = 1.  # confidence
        self.r_max = r_max

        # For SMDP
        self.tau = 0.9
        self.tau_max = 1.0
        self.tau_min = 1.0

    def _update_stats(self, s, a, r, terminal, ns):
        s_id = self.representation.state_id(s)
        scale_f = self.n_obs[s_id, a] + self.n_k[s_id, a]
        # Update reward esitimator
        r_hat = self.estimated_rewards[s_id, a]
        self.estimated_rewards[s_id, a] = (r_hat * scale_f + r) / (scale_f + 1.0)

        ns_id = self.representation.state_id(ns)
        self.counter[s_id, a, ns_id] += 1
        self.visited_sa.add((s_id, a))
        self.n_k[s, a] += 1

        self.iteration += 1

    def _beta_r(self):
        log_c = 2.0 * self.n_states * self.n_actions
        ci = chernoff(self.iteration, self.n_obs, self.delta, 3.5, log_c)
        return ci * self.r_max

    def _beta_p(self):
        s, a = self.n_states, self.n_actions
        beta = chernoff(self.iteration, self.n_obs, self.delta, 14.0 * s, 2.0 * a)
        return self.alpha_p * beta.reshape((s, a, 1))

    def _beta_tau(self):
        """
        Estimated holding time(only for SMDP)
        """
        return np.zeros_like(self.estimated_holding_times)

    def show_dim(self, *args):
        for arg in args:
            if isinstance(arg, np.ndarray):
                print(arg.shape)

    def _solve_optimisitic_mdp(self):
        beta_r = self._beta_r()
        beta_tau = self._beta_tau()
        beta_p = self._beta_p()
        return self.opt_solver.run(
            self.solver_policy_indices,
            self.solver_policy,
            self.prob,
            self.estimated_rewards,
            self.estimated_holding_times,
            beta_r,
            beta_p,
            beta_tau,
            self.tau_max,
            self.r_max,
            self.tau,
            self.tau_min,
            self.r_max / np.sqrt(self.iteration + 1.0),
        )

    def _init_episode(self):
        self.n_k.fill(0)
        self.delta = 1.0 / np.sqrt(self.iteration + 1.0)
        span_value = self._solve_optimisitic_mdp()

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self._update_stats(s, a, r, terminal, ns)
        if terminal is False:
            return
        self._init_episode()
