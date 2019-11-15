"""Least-Squares Policy Iteration [Lagoudakis and Parr 2003]."""
import numpy as np
from rlpy import tools
from scipy import sparse as sp
from .batch_agent import BatchAgent

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


class LSPI(BatchAgent):
    """Least Squares Policy Iteration reinforcement learning agent.
    """

    def __init__(
        self,
        policy,
        representation,
        discount_factor,
        max_window,
        steps_between_lspi,
        lspi_iterations=5,
        tol_epsilon=1e-3,
        re_iterations=100,
        use_sparse=False,
    ):
        """
        :param representation: Representation over the state features used by the agent.
        :param policy: Policy used by the agent.
        :param discount_factor: discount factor of future rewards
        :param max_window): Maximum number of steps the agent will be run for,
            which acts as the number of transitions to store.
        :param steps_between_lspi: Number of steps between runs of the lspi algorithm.
        :param lspi_iterations: Maximum number of iterations to go through
            for each update with LSPI.
        :param tol_epsilond: Stopping criteria/threshold  for when LSPI is
            considered to have converged.
        :param re_iterations: Number of iterations of representation expansion to run.
        :param use_sparse: Use sparse operators for building the matrix of transitions?
        """
        super().__init__(policy, representation, discount_factor, max_window)
        self.steps_between_lspi = steps_between_lspi
        self.tol_epsilon = tol_epsilon
        self.lspi_iterations = lspi_iterations
        self.re_iterations = re_iterations
        self.use_sparse = use_sparse

        # Make A and r incrementally if the representation can not expand
        self.fixed_rep = not representation.IS_DYNAMIC
        if self.fixed_rep:
            f_size = representation.features_num * representation.actions_num
            self.b = np.zeros((f_size, 1))
            self.A = np.zeros((f_size, f_size))

            # Cache calculated phi vectors
            if self.use_sparse:
                self.all_phi_s = sp.lil_matrix(
                    (max_window, representation.features_num)
                )
                self.all_phi_ns = sp.lil_matrix(
                    (max_window, representation.features_num)
                )
                self.all_phi_s_a = sp.lil_matrix((max_window, f_size))
                self.all_phi_ns_na = sp.lil_matrix((max_window, f_size))
            else:
                self.all_phi_s = np.zeros((max_window, representation.features_num))
                self.all_phi_ns = np.zeros((max_window, representation.features_num))
                self.all_phi_s_a = np.zeros((max_window, f_size))
                self.all_phi_ns_na = np.zeros((max_window, f_size))

    def calculateTDErrors(self):
        """Calculate TD errors over the transition instances stored.
        Returns the TD errors for all transitions with the current parameters.
        """
        # Calculates the TD-Errors in a matrix format for a set of samples = R
        # + (discount_factor*F2 - F1) * Theta
        discount_factor = self.discount_factor
        R = self.data_r[: self.samples_count, :]
        if self.use_sparse:
            F1 = sp.csr_matrix(self.all_phi_s_a[: self.samples_count, :])
            F2 = sp.csr_matrix(self.all_phi_ns_na[: self.samples_count, :])
            answer = R + (
                discount_factor * F2 - F1
            ) * self.representation.weight_vec.reshape(-1, 1)
            return np.squeeze(np.asarray(answer))
        else:
            F1 = self.all_phi_s_a[: self.samples_count, :]
            F2 = self.all_phi_ns_na[: self.samples_count, :]
            return R.ravel() + np.dot(
                discount_factor * F2 - F1, self.representation.weight_vec
            )

    def episode_terminated(self):
        """This function adjusts all necessary elements of the agent at the end of
        the episodes.
        .. note::
            Every agent must call this function at the end of the learning if the
            transition led to terminal state.
        """
        # Increase the number of episodes
        self.episode_count += 1

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        """Iterative learning method for the agent.

        :param ndarray s: The current state features.
        :param ndarray p_actions: The actions available in state s.
        :param int a: The action taken by the agent in state s.
        :param float r: The reward received by the agent for taking action a in state s.
        :param ndarray ns: The next state features.
        :param ndarray np_actions: The actions available in state ns.
        :param int na: The action taken by the agent in state ns.
        :param bool terminal: Whether or not ns is a terminal state.
        """
        super().learn(s, p_actions, a, r, ns, np_actions, na, terminal)
        if self.samples_count % self.steps_between_lspi == 0:
            self.representationExpansionLSPI()

    def representationExpansionLSPI(self):
        re_iteration = 0
        added_feature = True

        if self.representation.features_num == 0:
            print("No features, hence no LSPI is necessary!")
            return

        self.logger.info(
            "============================\nRunning LSPI with %d Samples\n============================"
            % self.samples_count
        )
        while added_feature and re_iteration <= self.re_iterations:
            re_iteration += 1
            # Some Prints
            if tools.hasFunction(self.representation, "batch_discover"):
                self.logger.info(
                    "-----------------\nRepresentation Expansion iteration #%d\n-----------------"
                    % re_iteration
                )
            # Run LSTD for first solution
            self.LSTD()
            # Run Policy Iteration to change a_prime and recalculate weight_vec in a
            # loop
            td_errors = self.policyIteration()
            # Add new Features
            if tools.hasFunction(self.representation, "batch_discover"):
                added_feature = self.representation.batch_discover(
                    td_errors,
                    self.all_phi_s[: self.samples_count, :],
                    self.data_s[: self.samples_count, :],
                )
            else:
                added_feature = False
            # print 'L_inf distance to V*= ',
        if added_feature:
            # Run LSPI one last time with the new features
            self.LSTD()
            self.policyIteration()

    def policyIteration(self):
        """Update the policy by recalculating A based on new na.

        Returns the TD error for each sample based on the latest weights and
        next actions.
        """
        start_time = tools.clock()
        weight_diff = self.tol_epsilon + 1  # So that the loop starts
        lspi_iteration = 0
        self.best_performance = -np.inf
        self.logger.info("Running Policy Iteration:")

        # We save action_mask on the first iteration (used for batch_best_action) to
        # reuse it and boost the speed
        # action_mask is a matrix that shows which actions are available for each state
        action_mask = None
        discount_factor = self.discount_factor
        F1 = (
            sp.csr_matrix(self.all_phi_s_a[: self.samples_count, :])
            if self.use_sparse
            else self.all_phi_s_a[: self.samples_count, :]
        )
        while lspi_iteration < self.lspi_iterations and weight_diff > self.tol_epsilon:

            # Find the best action for each state given the current value function
            # Notice if actions have the same value the first action is
            # selected in the batch mode
            iteration_start_time = tools.clock()
            (
                best_action,
                self.all_phi_ns_new_na,
                action_mask,
            ) = self.representation.batch_best_action(
                self.data_ns[: self.samples_count, :],
                self.all_phi_ns,
                action_mask,
                self.use_sparse,
            )

            # Recalculate A matrix (b remains the same)
            # Solve for the new weight_vec
            if self.use_sparse:
                F2 = sp.csr_matrix(self.all_phi_ns_new_na[: self.samples_count, :])
                A = F1.T * (F1 - discount_factor * F2)
            else:
                F2 = self.all_phi_ns_new_na[: self.samples_count, :]
                A = np.dot(F1.T, F1 - discount_factor * F2)

            A = tools.regularize(A)
            new_weight_vec, solve_time = tools.solveLinear(A, self.b)

            # Calculate TD_Errors
            ####################
            td_errors = self.calculateTDErrors()

            # Calculate the weight difference. If it is big enough update the
            # weight_vec
            weight_diff = np.linalg.norm(
                self.representation.weight_vec - new_weight_vec
            )
            if weight_diff > self.tol_epsilon:
                self.representation.weight_vec = new_weight_vec

            self.logger.info(
                "%d: %0.0f(s), ||w1-w2|| = %0.4f, Sparsity=%0.1f%%, %d Features"
                % (
                    lspi_iteration + 1,
                    tools.deltaT(iteration_start_time),
                    weight_diff,
                    tools.sparsity(A),
                    self.representation.features_num,
                )
            )
            lspi_iteration += 1

        self.logger.info(
            "Total Policy Iteration Time = %0.0f(s)" % tools.deltaT(start_time)
        )
        return td_errors

    def LSTD(self):
        """Run the LSTD algorithm on the collected data, and update the
        policy parameters.
        """
        start_time = tools.clock()

        if not self.fixed_rep:
            # build phi_s and phi_ns for all samples
            p = self.samples_count
            n = self.representation.features_num
            self.all_phi_s = np.empty((p, n), dtype=self.representation.feature_type())
            self.all_phi_ns = np.empty((p, n), dtype=self.representation.feature_type())

            for i in np.arange(self.samples_count):
                self.all_phi_s[i, :] = self.representation.phi(self.data_s[i])
                self.all_phi_ns[i, :] = self.representation.phi(self.data_ns[i])

            # build phi_s_a and phi_ns_na for all samples given phi_s and
            # phi_ns
            self.all_phi_s_a = self.representation.batch_phi_sa(
                self.all_phi_s[: self.samples_count, :],
                self.data_a[: self.samples_count, :],
                use_sparse=self.use_sparse,
            )
            self.all_phi_ns_na = self.representation.batch_phi_sa(
                self.all_phi_ns[: self.samples_count, :],
                self.data_na[: self.samples_count, :],
                use_sparse=self.use_sparse,
            )

            # calculate A and b for LSTD
            F1 = self.all_phi_s_a[: self.samples_count, :]
            F2 = self.all_phi_ns_na[: self.samples_count, :]
            R = self.data_r[: self.samples_count, :]
            discount_factor = self.discount_factor

            if self.use_sparse:
                self.b = (F1.T * R).reshape(-1, 1)
                self.A = F1.T * (F1 - discount_factor * F2)
            else:
                self.b = np.dot(F1.T, R).reshape(-1, 1)
                self.A = np.dot(F1.T, F1 - discount_factor * F2)

        A = tools.regularize(self.A)

        # Calculate weight_vec
        self.representation.weight_vec, solve_time = tools.solveLinear(A, self.b)

        # log solve time only if takes more than 1 second
        if solve_time > 1:
            self.logger.info(
                "Total LSTD Time = %0.0f(s), Solve Time = %0.0f(s)"
                % (tools.deltaT(start_time), solve_time)
            )
        else:
            self.logger.info("Total LSTD Time = %0.0f(s)" % (tools.deltaT(start_time)))

    def store_samples(self, s, a, r, ns, na, terminal):
        """Process one transition instance."""

        # Update A and b if representation is going to be fix together with all
        # features
        if self.fixed_rep:
            if terminal:
                phi_s = self.representation.phi(s, False)
                phi_s_a = self.representation.phi_sa(s, False, a, phi_s=phi_s)
            else:
                # This is because the current s,a will be the previous ns, na
                if self.use_sparse:
                    phi_s = self.all_phi_ns[self.samples_count - 1, :].todense()
                    phi_s_a = self.all_phi_ns_na[self.samples_count - 1, :].todense()
                else:
                    phi_s = self.all_phi_ns[self.samples_count - 1, :]
                    phi_s_a = self.all_phi_ns_na[self.samples_count - 1, :]

            phi_ns = self.representation.phi(ns, terminal)
            phi_ns_na = self.representation.phi_sa(ns, terminal, na, phi_s=phi_ns)

            self.all_phi_s[self.samples_count, :] = phi_s
            self.all_phi_ns[self.samples_count, :] = phi_ns
            self.all_phi_s_a[self.samples_count, :] = phi_s_a
            self.all_phi_ns_na[self.samples_count, :] = phi_ns_na

            discount_factor = self.discount_factor
            self.b += phi_s_a.reshape((-1, 1)) * r
            d = phi_s_a - discount_factor * phi_ns_na
            self.A += np.outer(phi_s_a, d)

            super().store_samples(s, a, r, ns, na, terminal)
