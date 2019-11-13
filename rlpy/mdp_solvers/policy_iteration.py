"""Classical Policy Iteration.
Performs Bellman Backup on a given s,a pair given a fixed policy by sweeping through the
state space. Once the errors are bounded, the policy is changed.
"""
from copy import deepcopy
import numpy as np
from rlpy.policies import eGreedy
from rlpy.tools import deltaT, hhmmss, clock, l_norm
from .mdp_solver import MDPSolver

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


class PolicyIteration(MDPSolver):
    """
    Policy Iteration MDP Solver.
    """

    def __init__(self, *args, max_pe_iterations=10, **kwargs):
        """
        :param max_pe_iterations: Maximum number of Policy evaluation iterations to run.
        """
        super().__init__(*args, **kwargs)
        self.max_pe_iterations = max_pe_iterations
        self.bellman_updates = 0
        self.logger.info("Max PE Iterations:\t%d" % self.max_pe_iterations)

        if not self.is_tabular():
            raise ValueError(
                "Policy Iteration works only with a tabular representation."
            )

    def policy_evaluation(self, policy):
        """
        Evaluate a given policy: this is done by applying the Bellman backup over all
        states until the change is less than a given threshold.

        Returns: convergence status as a boolean
        """
        converged = False
        policy_evaluation_iteration = 0
        while (
            not converged
            and self.has_time()
            and policy_evaluation_iteration < self.max_pe_iterations
        ):
            policy_evaluation_iteration += 1

            # Sweep The State Space
            for i in range(0, self.representation.agg_states_num):

                # Check for solver time
                if not self.has_time():
                    break

                # Map an state ID to state
                s = self.representation.stateID2state(i)

                # Skip terminal states and states with no possible action
                possible_actions = self.domain.possible_actions(s=s)
                if self.domain.is_terminal(s) or len(possible_actions) == 0:
                    continue

                # Apply Bellman Backup
                self.bellman_backup(
                    s, policy.pi(s, False, possible_actions), self.ns_samples, policy
                )

                # Update number of backups
                self.bellman_updates += 1

                # Check for the performance
                if self.bellman_updates % self.log_interval == 0:
                    performance_return = self.performance_run()[0]
                    self.logger.info(
                        "[%s]: BellmanUpdates=%d, Return=%0.4f"
                        % (
                            hhmmss(deltaT(self.start_time)),
                            self.bellman_updates,
                            performance_return,
                        )
                    )

            # check for convergence: L_infinity norm of the difference between the to
            # the weight vector of representation
            weight_diff = l_norm(
                policy.representation.weight - self.representation.weight
            )
            converged = weight_diff < self.convergence_threshold

            # Log Status
            self.logger.info(
                "PE #%d [%s]: BellmanUpdates=%d, ||delta-weight_vec||=%0.4f"
                % (
                    policy_evaluation_iteration,
                    hhmmss(deltaT(self.start_time)),
                    self.bellman_updates,
                    weight_diff,
                )
            )

            # Show Plots
            if self._visualize_mode:
                self.domain.show_learning(self.representation)
        return converged

    def policy_improvement(self, policy):
        """
        Given a policy improve it by taking the greedy action in each state based
        on the value function.
        Returns the new policy.
        """
        policyChanges = 0
        i = 0
        while i < self.representation.agg_states_num and self.has_time():
            s = self.representation.stateID2state(i)
            p_actions = self.domain.possible_actions(s)
            if not self.domain.is_terminal(s) and len(self.domain.possible_actions(s)):
                for a in self.domain.possible_actions(s):
                    self.bellman_backup(s, a, self.ns_samples, policy)
                p_actions = self.domain.possible_actions(s=s)
                best_action = self.representation.best_action(s, False, p_actions)
                if policy.pi(s, False, p_actions) != best_action:
                    policyChanges += 1
            i += 1

        # This will cause the policy to be copied over
        policy.representation.weight = self.representation.weight.copy()
        perf_return, perf_steps, perf_term, perf_disc_return = self.performance_run()
        self.logger.info(
            "PI #%d [%s]: BellmanUpdates=%d, Policy Change=%d, Return=%0.4f, Steps=%d"
            % (
                self.policy_improvement_iteration,
                hhmmss(deltaT(self.start_time)),
                self.bellman_updates,
                policyChanges,
                perf_return,
                perf_steps,
            )
        )

        # store stats
        self.result["bellman_updates"].append(self.bellman_updates)
        self.result["return"].append(perf_return)
        self.result["planning_time"].append(deltaT(self.start_time))
        self.result["num_features"].append(self.representation.features_num)
        self.result["steps"].append(perf_steps)
        self.result["terminated"].append(perf_term)
        self.result["discounted_return"].append(perf_disc_return)
        self.result["policy_improvement_iteration"].append(
            self.policy_improvement_iteration
        )
        return policy, policyChanges

    def _solve_impl(self):
        """Solve the domain MDP."""
        self.bellman_updates = 0
        self.policy_improvement_iteration = 0
        self.start_time = clock()

        # Initialize the policy
        # Copy the representation so that the weight change during the evaluation
        # does not change the policy
        policy = eGreedy(deepcopy(self.representation), epsilon=0, deterministic=True)

        # Setup the number of policy changes to 1 so the while loop starts
        policy_changes = True

        while policy_changes and self.has_time():
            # Evaluate the policy
            if self.policy_evaluation(policy):
                self.logger.info("Converged!")

            # Improve the policy
            self.policy_improvement_iteration += 1
            policy, policy_changes = self.policy_improvement(policy)

        self.log_value()
