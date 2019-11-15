"""Trajectory Based Value Iteration. This algorithm is different from Value iteration
   in 2 senses:
     1. It works with any Linear Function approximator
     2. Samples are gathered using the e-greedy policy

The algorithm terminates if the maximum bellman-error in a consequent set of
trajectories is below a threshold
"""
from .mdp_solver import MDPSolver
from rlpy.tools import deltaT, hhmmss, clock
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
__author__ = "Alborz Geramifard"


class TrajectoryBasedValueIteration(MDPSolver):
    """Trajectory Based Value Iteration MDP Solver.
    """

    # Minimum number of trajectories required for convergence in which the max
    # bellman error was below the threshold
    MIN_CONVERGED_TRAJECTORIES = 5

    def __init__(self, *args, alpha=0.1, epsilon=0.1, **kwargs):
        """
        :param alpha: Step size parameter to adjust the weights. If the representation
            is tabular, you can set this to 1.
        :param epsilon: Probability of taking a random action during each
            decision making.
        """
        super().__init__(*args, **kwargs)

        self.epsilon = epsilon
        if self.is_tabular():
            self.alpha = 1
        else:
            self.alpha = alpha

    def eps_greedy(self, s, terminal, p_actions):
        if self.random_state.rand() > self.epsilon:
            return self.representation.best_action(s, terminal, p_actions)
        else:
            return self.random_state.choice(p_actions)

    def _solve_impl(self):
        """Solve the domain MDP."""

        # Used to show the total time took the process
        self.start_time = clock()
        bellman_updates = 0
        converged = False
        iteration = 0
        # Track the number of consequent trajectories with very small observed
        # BellmanError
        converged_trajectories = 0
        while self.has_time() and not converged:
            max_bellman_error = 0
            step = 0
            s, terminal, p_actions = self.domain.s0()
            # Generate a new episode e-greedy with the current values
            while not terminal and step < self.domain.episode_cap and self.has_time():
                a = self.eps_greedy(s, terminal, p_actions)
                bellman_error, phi_s, phi_s_a = self._bellman_error(s, a, terminal)
                # Update Parameters
                self.representation.weight_vec += self.alpha * bellman_error * phi_s_a
                bellman_updates += 1
                step += 1

                # Discover features if the representation has the discover method
                if hasattr(self.representation, "discover"):
                    self.representation.post_discover(phi_s, bellman_error)

                max_bellman_error = max(max_bellman_error, abs(bellman_error))
                # Simulate new state and action on trajectory
                _, s, terminal, p_actions = self.domain.step(a)

            # check for convergence
            iteration += 1
            if max_bellman_error < self.convergence_threshold:
                converged_trajectories += 1
            else:
                converged_trajectories = 0
            (
                perf_return,
                perf_steps,
                perf_term,
                perf_disc_return,
            ) = self.performance_run()
            converged = converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES
            self.logger.info(
                "PI #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Return=%0.4f,"
                "Steps=%d, Features=%d"
                % (
                    iteration,
                    hhmmss(deltaT(self.start_time)),
                    bellman_updates,
                    max_bellman_error,
                    perf_return,
                    perf_steps,
                    self.representation.features_num,
                )
            )
            if self._visualize_mode:
                self.domain.show_learning(self.representation)

            # store stats
            self.result["bellman_updates"].append(bellman_updates)
            self.result["return"].append(perf_return)
            self.result["planning_time"].append(deltaT(self.start_time))
            self.result["num_features"].append(self.representation.features_num)
            self.result["steps"].append(perf_steps)
            self.result["terminated"].append(perf_term)

            self.result["discounted_return"].append(perf_disc_return)
            self.result["iteration"].append(iteration)

        if converged:
            self.logger.info("Converged!")
        self.log_value()
