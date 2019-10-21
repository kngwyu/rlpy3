"""Greedy-GQ(lambda) learning agent"""
import numpy as np
from rlpy.tools import add_new_features, count_nonzero
from .agent import Agent, DescentAlgorithm

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


class GreedyGQ(Agent, DescentAlgorithm):
    """
    A variant of Q-learning for use with linear function approximation.
    Convergence guarantees are available even without an exact value
    function representation.
    See Maei et al., 2010 (http://www.icml2010.org/papers/627.pdf)

    """

    def __init__(
        self,
        policy,
        representation,
        discount_factor=1.0,
        lambda_=0,
        beta_coef=1e-6,
        **kwargs
    ):
        super().__init__(policy, representation, discount_factor)
        DescentAlgorithm.__init__(self, **kwargs)
        self.eligibility_trace = np.zeros(
            representation.features_num * representation.actions_num
        )
        # use a state-only version of eligibility trace for dabney decay mode
        self.eligibility_trace_s = np.zeros(representation.features_num)
        self.lambda_ = lambda_
        self.gqweight = self.representation.weight_vec.copy()
        # The beta in the GQ algorithm is assumed to be learn_rate * THIS CONSTANT
        self.second_lr_coef = beta_coef

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self.representation.pre_discover(s, False, a, ns, terminal)
        discount_factor = self.discount_factor
        weight_vec = self.representation.weight_vec
        phi_s = self.representation.phi(s, False)
        phi = self.representation.phi_sa(s, False, a, phi_s)
        phi_prime_s = self.representation.phi(ns, terminal)
        # Switch na to the best possible action
        na = self.representation.best_action(ns, terminal, np_actions, phi_prime_s)
        phi_prime = self.representation.phi_sa(ns, terminal, na, phi_prime_s)
        nnz = count_nonzero(phi_s)  # Number of non-zero elements

        expanded = (len(phi) - len(self.gqweight)) // self.representation.actions_num
        if expanded > 0:
            self._expand_vectors(expanded)
        # Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace *= discount_factor * self.lambda_
            self.eligibility_trace += phi

            self.eligibility_trace_s *= discount_factor * self.lambda_
            self.eligibility_trace_s += phi_s

            # Set max to 1
            self.eligibility_trace[self.eligibility_trace > 1] = 1
            self.eligibility_trace_s[self.eligibility_trace_s > 1] = 1
        else:
            self.eligibility_trace = phi
            self.eligibility_trace_s = phi_s

        td_error = r + np.dot(discount_factor * phi_prime - phi, weight_vec)

        self.updateLearnRate(
            phi_s, phi_prime_s, self.eligibility_trace_s, discount_factor, nnz, terminal
        )

        if nnz > 0:  # Phi has some nonzero elements, proceed with update
            td_error_estimate_now = np.dot(phi, self.gqweight)
            delta_weight_vec = (
                td_error * self.eligibility_trace
                - discount_factor * td_error_estimate_now * phi_prime
            )
            weight_vec += self.learn_rate * delta_weight_vec
            delta_gqweight = (td_error - td_error_estimate_now) * phi
            self.gqweight += self.learn_rate * self.second_lr_coef * delta_gqweight

        expanded = self.representation.post_discover(s, False, a, td_error, phi_s)
        if expanded:
            self._expand_vectors(expanded)
        if terminal:
            self.episode_terminated()

    def _expand_vectors(self, num_expansions):
        """
        correct size of GQ weight and e-traces when new features were expanded
        """
        new_elem = np.zeros((self.representation.actions_num, num_expansions))
        self.gqweight = add_new_features(self.gqweight, new_elem)
        if self.lambda_:
            # Correct the size of eligibility traces (pad with zeros for new
            # features)
            self.eligibility_trace = add_new_features(
                self.eligibility_trace, self.representation.actions_num, new_elem
            )
            self.eligibility_trace_s = add_new_features(
                self.eligibility_trace_s, np.zeros((1, num_expansions))
            )
