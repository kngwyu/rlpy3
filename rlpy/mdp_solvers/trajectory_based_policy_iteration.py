"""Trajectory Based Policy Iteration:
    Loop until the weight change to the value function is small for some number of
    trajectories
    (cant check policy because we dont store anything in the size of the state-space)
    1. Update the evaluation of the policy till the change is small.
    2. Update the policy

    * There is solve_in_matrix_format function which does policy evaluation in one shot
      using samples collected in the matrix format.
      Since the algorithm toss out the samples, convergence is hardly reached
      because the policy may alternate.
"""
from .mdp_solver import MDPSolver
from rlpy.tools import (
    add_new_features,
    hhmmss,
    deltaT,
    hasFunction,
    solveLinear,
    regularize,
    clock,
    l_norm,
)
from rlpy.policies import eGreedy
import numpy as np
from copy import deepcopy

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


class TrajectoryBasedPolicyIteration(MDPSolver):
    """Trajectory Based Policy Iteration MDP Solver.
    """

    # step size parameter to adjust the weights. If the representation is
    # tabular you can set this to 1.

    # Minimum number of trajectories required for convergence in which the max
    # bellman error was below the threshold
    MIN_CONVERGED_TRAJECTORIES = 5

    def __init__(self, *args, alpha=0.1, epsilon=0.1, max_pe_iterations=10, **kwargs):
        """
        :param alpha: Step size parameter to adjust the weights. If the representation
            is tabular, you can set this to 1.
        :param epsilon: Probability of taking a random action during each
            decision making.
        :param max_pe_iterations: Maximum number of Policy evaluation iterations to run.
        """
        super().__init__(*args, **kwargs)

        self.epsilon = epsilon
        self.max_pe_iterations = max_pe_iterations
        if self.is_tabular():
            self.alpha = 1
        else:
            self.alpha = alpha

    def sample_ns_na(self, policy, action=None, start_trajectory=False):
        """
        Given a policy sample the next state and next action along the trajectory
         followed by the policy
        * Noise is added in selecting action:
        with probability 1-e, follow the policy
        with probability self.epsilon pick a uniform random action from possible
        actions.
        * if start_trajectory = True the initial state is sampled from s0() function
        of the domain otherwise take the action given in the current state.
        """
        if start_trajectory:
            ns, terminal, possible_actions = self.domain.s0()
        else:
            _, ns, terminal, possible_actions = self.domain.step(action)

        if self.random_state.rand() > self.epsilon:
            na = policy.pi(ns, terminal, possible_actions)
        else:
            na = self.random_state.choice(possible_actions)

        return ns, na, terminal

    def traj_based_policy_evaluation(self, policy):
        """
        Evaluate the current policy by simulating trajectories and update
        the value function along the visited states.
        """
        PE_iteration = 0
        evaluation_is_accurate = False
        converged_trajectories = 0
        while (
            not evaluation_is_accurate
            and self.has_time()
            and PE_iteration < self.max_pe_iterations
        ):

            # Generate a new episode e-greedy with the current values
            max_bellman_error = 0
            step = 0

            s, a, terminal = self.sample_ns_na(policy, start_trajectory=True)

            while not terminal and step < self.domain.episode_cap and self.has_time():
                bellman_error, phi_s, phi_s_a = self._bellman_error(s, a, terminal)

                # Update the value function using approximate bellman backup
                self.representation.weight_vec += self.alpha * bellman_error * phi_s_a
                self.bellman_updates += 1
                step += 1
                max_bellman_error = max(max_bellman_error, abs(bellman_error))

                # Discover features if the representation has the discover method
                if hasattr(self.representation, "discover"):
                    self.representation.post_discover(phi_s, bellman_error)

                s, a, terminal = self.sample_ns_na(policy, a)

            # check for convergence of policy evaluation
            PE_iteration += 1
            if max_bellman_error < self.convergence_threshold:
                converged_trajectories += 1
            else:
                converged_trajectories = 0
            evaluation_is_accurate = (
                converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES
            )
            self.logger.info(
                "PE #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Features=%d"
                % (
                    PE_iteration,
                    hhmmss(deltaT(self.start_time)),
                    self.bellman_updates,
                    max_bellman_error,
                    self.representation.features_num,
                )
            )

    def _solve_impl(self):
        """Solve the domain MDP."""

        self.start_time = clock()  # Used to track the total time for solving
        self.bellman_updates = 0
        converged = False
        PI_iteration = 0

        # The policy is maintained as separate copy of the representation.
        # This way as the representation is updated the policy remains intact
        policy = eGreedy(deepcopy(self.representation), epsilon=0, deterministic=True)
        a_num = self.domain.actions_num

        while self.has_time() and not converged:

            # Policy Improvement (Updating the representation of the value)
            self.traj_based_policy_evaluation(policy)
            PI_iteration += 1

            # Theta can increase in size if the representation
            # is expanded hence padding the weight vector with zeros
            additional_dim = (
                self.representation.features_num - policy.representation.features_num
            )
            padded_theta = np.hstack(
                (policy.representation.weight, np.zeros((a_num, additional_dim)))
            )

            # Calculate the change in the weight_vec as L2-norm
            weight_diff = np.linalg.norm(padded_theta - self.representation.weight)
            converged = weight_diff < self.convergence_threshold

            # Update the underlying value function of the policy
            policy.representation = deepcopy(self.representation)  # self.representation

            (
                perf_return,
                perf_steps,
                perf_term,
                perf_disc_return,
            ) = self.performance_run()
            self.logger.info(
                "PI #%d [%s]: BellmanUpdates=%d, ||delta-weight_vec||=%0.4f, "
                "Return=%0.3f, steps=%d, features=%d"
                % (
                    PI_iteration,
                    hhmmss(deltaT(self.start_time)),
                    self.bellman_updates,
                    weight_diff,
                    perf_return,
                    perf_steps,
                    self.representation.features_num,
                )
            )

            if self._visualize_mode:
                self.domain.show_learning(self.representation)

            # store stats
            self.result["bellman_updates"].append(self.bellman_updates)
            self.result["return"].append(perf_return)
            self.result["planning_time"].append(deltaT(self.start_time))
            self.result["num_features"].append(self.representation.features_num)
            self.result["steps"].append(perf_steps)
            self.result["terminated"].append(perf_term)
            self.result["discounted_return"].append(perf_disc_return)
            self.result["policy_improvemnt_iteration"].append(PI_iteration)

        if converged:
            self.logger.info("Converged!")
        self.log_value()

    def solve_in_matrix_format(self):
        # while delta_weight_vec > threshold
        #  1. Gather data following an e-greedy policy
        #  2. Calculate A and b estimates
        #  3. calculate new_weight_vec, and delta_weight_vec
        # return policy greedy w.r.t last weight_vec
        self.policy = eGreedy(self.representation, epsilon=self.epsilon)

        # Number of samples to be used for each policy evaluation phase. L1 in
        # the Geramifard et. al. FTML 2012 paper
        self.samples_num = 1000

        self.start_time = clock()  # Used to track the total time for solving
        samples = 0
        converged = False
        iteration = 0
        while self.has_time() and not converged:

            #  1. Gather samples following an e-greedy policy
            S, Actions, NS, R, T = self.collect_samples(self.samples_num)
            samples += self.samples_num

            #  2. Calculate A and b estimates
            a_num = self.domain.actions_num
            n = self.representation.features_num
            discount_factor = self.domain.discount_factor

            self.A = np.zeros((n * a_num, n * a_num))
            self.b = np.zeros((n * a_num, 1))
            for i in range(self.samples_num):
                phi_s_a = self.representation.phi_sa(S[i], T[i], Actions[i, 0]).reshape(
                    (-1, 1)
                )
                E_phi_ns_na = self.calculate_expected_phi_ns_na(
                    S[i], Actions[i, 0], self.ns_samples
                ).reshape((-1, 1))
                d = phi_s_a - discount_factor * E_phi_ns_na
                self.A += np.outer(phi_s_a, d.T)
                self.b += phi_s_a * R[i, 0]

            #  3. calculate new_weight_vec, and delta_weight_vec
            new_weight_vec, solve_time = solveLinear(regularize(self.A), self.b)
            iteration += 1
            if solve_time > 1:
                self.logger.info(
                    "#%d: Finished Policy Evaluation. Solve Time = %0.2f(s)"
                    % (iteration, solve_time)
                )
            weight_diff = l_norm(new_weight_vec - self.representation.weight_vec)
            converged = weight_diff < self.convergence_threshold
            self.representation.weight_vec = new_weight_vec
            (
                perf_return,
                perf_steps,
                perf_term,
                perf_disc_return,
            ) = self.performance_run()
            self.logger.info(
                "#%d [%s]: Samples=%d, ||weight-Change||=%0.4f, Return = %0.4f"
                % (
                    iteration,
                    hhmmss(deltaT(self.start_time)),
                    samples,
                    weight_diff,
                    perf_return,
                )
            )
            if self._visualize_mode:
                self.domain.show_learning(self.representation)

            # store stats
            self.result["samples"].append(samples)
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

    def calculate_expected_phi_ns_na(self, s, a, ns_samples):
        # calculate the expected next feature vector (phi(ns,pi(ns)) given s
        # and a. Eqns 2.20 and 2.25 in [Geramifard et. al. 2012 FTML Paper]
        if hasFunction(self.domain, "expected_step"):
            p, r, ns, t, pa = self.domain.expected_step(s, a)
            phi_ns_na = np.zeros(
                self.representation.features_num * self.domain.actions_num
            )
            for j, pj in enumerate(p):
                na = self.policy.pi(ns[j], t[j], pa[j])
                phi_ns_na += pj * self.representation.phi_sa(ns[j], t[j], na)
        else:
            next_states, rewards = self.domain.sampleStep(s, a, ns_samples)
            phi_ns_na = np.mean(
                [
                    self.representation.phisa(ns, self.policy.pi(ns))
                    for ns in next_states
                ]
            )
        return phi_ns_na
