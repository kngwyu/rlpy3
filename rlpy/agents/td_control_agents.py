"""Control agents based on TD Learning, i.e., Q-Learning and SARSA"""
from .agent import Agent, DescentAlgorithm
from rlpy.tools import add_new_features, count_nonzero
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


class TDControlAgent(Agent, DescentAlgorithm):
    """
    abstract class for the control variants of the classical linear TD-Learning.
    It is the parent of SARSA and Q-Learning

    All children must implement the _future_action function.
    """

    def __init__(self, policy, representation, discount_factor, lambda_=0, **kwargs):
        Agent.__init__(self, policy, representation, discount_factor)
        DescentAlgorithm.__init__(self, **kwargs)
        self.eligibility_trace = np.zeros(
            representation.features_num * representation.actions_num
        )
        #: lambda Parameter in SARSA [Sutton Book 1998]
        self.lambda_ = lambda_

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        """needs to be implemented by children"""
        pass

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        # The previous state could never be terminal
        # (otherwise the episode would have already terminated)
        prevStateTerminal = False

        self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)
        discount_factor = self.discount_factor
        phi_s = self.representation.phi(s, prevStateTerminal)
        phi = self.representation.phi_sa(s, prevStateTerminal, a, phi_s)
        phi_prime_s = self.representation.phi(ns, terminal)
        # here comes the difference between SARSA and Q-Learning
        na = self._future_action(ns, terminal, np_actions, phi_prime_s, na)
        phi_prime = self.representation.phi_sa(ns, terminal, na, phi_prime_s)
        nnz = count_nonzero(phi_s)  # Number of non-zero elements

        # Set eligibility traces:
        if self.lambda_ > 0:
            expanded = (
                phi.shape[0] - self.eligibility_trace.shape[0]
            ) // self.representation.actions_num
            if expanded > 0:
                # Correct the size of eligibility traces (pad with zeros for
                # new features)
                self.eligibility_trace = add_new_features(
                    self.eligibility_trace,
                    np.zeros((self.representation.actions_num, expanded)),
                )

            self.eligibility_trace *= discount_factor * self.lambda_
            self.eligibility_trace += phi

            # Set max to 1
            self.eligibility_trace[self.eligibility_trace > 1] = 1
        else:
            self.eligibility_trace = phi

        td_error = r + np.dot(
            discount_factor * phi_prime - phi, self.representation.weight_vec
        )
        if nnz > 0:
            self.updateLearnRate(
                phi, phi_prime, self.eligibility_trace, discount_factor, nnz, terminal
            )
            weight_old = self.representation.weight.copy()
            self.representation.weight_vec += (
                self.learn_rate
                * self.representation.feature_learning_rate()
                * td_error
                * self.eligibility_trace
            )
            if not np.all(np.isfinite(self.representation.weight_vec)):
                self.representation.weight = weight_old
                import warnings

                warnings.warn(
                    "WARNING: TD-Learning diverged, weight_vec reached infinity!"
                )
        # Discover features if the representation has the discover method
        expanded = self.representation.post_discover(
            s, prevStateTerminal, a, td_error, phi_s
        )

        if terminal:
            # If THIS state is terminal:
            self.episode_terminated()


class Q_Learning(TDControlAgent):
    """
    The off-policy variant known as Q-Learning
    """

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        """Q Learning chooses the optimal action"""
        return self.representation.best_action(ns, terminal, np_actions, ns_phi)


class SARSA(TDControlAgent):
    """
    The on-policy variant known as SARSA.
    """

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        """SARS-->A<--, so SARSA simply chooses the action the agent will follow"""
        return na
