"""MDP Solver base class."""
from abc import ABC, abstractmethod
import numpy as np
import logging
from rlpy.tools import checkNCreateDirectory, className, deltaT, vec2id
from rlpy.tools.encoders import NpAwareEncoder
from collections import defaultdict
import os
import json

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "N. Kemal Ure"


class MDPSolver(ABC):
    """MDPSolver is the base class for model based reinforcement learning agents and
    planners.
    """

    # Maximum number of runs of an algorithm for averaging
    MAX_RUNS = 100

    def __init__(
        self,
        job_id,
        representation,
        domain,
        planning_time=np.inf,
        convergence_threshold=0.005,
        ns_samples=100,
        project_path=".",
        log_interval=5000,
    ):
        """
        :param job_id: The job id of this run of the algorithm(=random seed).
        :param represenation:  Link to the representation object.
        :param domain: Link to the domain object
        :param planning_time: Amount of time in seconds provided for the solver.
            After this it returnsits performance.
        :param convergence_threshold: Threshold to determine the convergence
            of the planner.
        :param nc_samples: Number of samples to be used to generate estimated bellman
            backup if the domain does not provide explicit probabilities
            though expected_step function.
        :param project_path: The place to save stats.
        :param log_interval: Number of bellman backups before reporting
            the performance. Not all planners may use this.
        """
        self.exp_id = job_id
        self.representation = representation
        self.domain = domain
        self.logger = logging.getLogger("rlpy.mdp_solvers." + self.__class__.__name__)
        self.ns_samples = ns_samples
        self.planning_time = planning_time
        self.project_path = project_path
        self.log_interval = log_interval
        self.convergence_threshold = convergence_threshold
        self._visualize_mode = False

        self.random_state = np.random.RandomState(seed=job_id)

        # TODO setup logging to file in experiment

        # create a dictionary of results
        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id
        self.output_filename = "{:0>3}-results.json".format(self.exp_id)

    @abstractmethod
    def _solve_impl(self):
        """Solve the domain MDP."""
        pass

    def solve(self, visualize=False):
        """Solve the domain MDP."""
        vis_orig = self._visualize_mode
        self._visualize_mode = visualize
        self._solve_impl()
        self._visualize_mode = vis_orig

    def log_value(self):
        self.logger.info(
            "Value of S0 is = %0.5f" % self.representation.V(*self.domain.s0())
        )
        self.save_stats()

    def bellman_backup(self, s, a, ns_samples, policy=None):
        """Applied Bellman Backup to state-action pair s,a
        i.e. Q(s,a) = E[r + discount_factor * V(s')]
        If policy is given then Q(s,a) =  E[r + discount_factor * Q(s',pi(s')]

        Args:
            s (ndarray):        The current state
            a (int):            The action taken in state s
            ns_samples(int):    Number of next state samples to use.
            policy (Policy):    Policy object to use for sampling actions.
        """
        Q = self.representation.q_look_ahead(s, a, ns_samples, policy)
        s_index = vec2id(
            self.representation.bin_state(s), self.representation.bins_per_dim
        )
        self.representation.weight[a, s_index] = Q

    def _bellman_error(self, s, a, terminal):
        new_Q = self.representation.q_look_ahead(s, a, self.ns_samples)
        phi_s = self.representation.phi(s, terminal)
        phi_s_a = self.representation.phi_sa(s, terminal, a, phi_s)
        old_Q = np.dot(phi_s_a, self.representation.weight_vec)
        return new_Q - old_Q, phi_s, phi_s_a

    def performance_run(self, visualize=False):
        """Set Exploration to zero and sample one episode from the domain."""

        eps_length = 0
        eps_return = 0
        eps_term = False
        eps_discounted_return = 0

        s, eps_term, p_actions = self.domain.s0()

        while not eps_term and eps_length < self.domain.episode_cap:
            a = self.representation.best_action(s, eps_term, p_actions)
            if visualize:
                self.domain.show_domain(a)
            r, ns, eps_term, p_actions = self.domain.step(a)
            s = ns
            eps_discounted_return += self.domain.discount_factor ** eps_length * r
            eps_return += r
            eps_length += 1
        if visualize:
            self.domain.show_domain(a)
        return eps_return, eps_length, eps_term, eps_discounted_return

    def save_stats(self):
        fullpath_output = os.path.join(self.project_path, self.output_filename)
        checkNCreateDirectory(self.project_path + "/")
        with open(fullpath_output, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True, cls=NpAwareEncoder)
        print("Saved in ", fullpath_output)

    def has_time(self):
        """Return a boolean stating if there is time left for planning."""
        return deltaT(self.start_time) < self.planning_time

    def is_tabular(self):
        """
        Check to see if the representation is Tabular as Policy Iteration and
        Value Iteration only work with Tabular representation.
        """
        return className(self.representation) == "Tabular"

    def collect_samples(self, samples):
        """
        Return matrices of S,A,NS,R,T where each row of each numpy 2d-array
        is a sample by following the current policy.

        - S: (#samples) x (# state space dimensions)
        - A: (#samples) x (1) int [we are storing actionIDs here, integers]
        - NS:(#samples) x (# state space dimensions)
        - R: (#samples) x (1) float
        - T: (#samples) x (1) bool

        See :py:meth:`~rlpy.agents.agent.Agent.Q_MC` and
            :py:meth:`~rlpy.agents.agent.Agent.MC_episode`.
        """
        domain = self.representation.domain
        S = np.empty(
            (samples, self.representation.domain.state_space_dims),
            dtype=type(domain.s0()),
        )
        A = np.empty((samples, 1), dtype="uint16")
        NS = S.copy()
        T = A.copy()
        R = np.empty((samples, 1))

        sample = 0
        eps_length = 0
        # So the first sample forces initialization of s and a
        terminal = True
        while sample < samples:
            if terminal or eps_length > self.representation.domain.episode_cap:
                s, terminal, possible_actions = domain.s0()
                a = self.policy.pi(s, terminal, possible_actions)

            # Transition
            r, ns, terminal, possible_actions = domain.step(a)
            # Collect Samples
            S[sample] = s
            A[sample] = a
            NS[sample] = ns
            T[sample] = terminal
            R[sample] = r

            sample += 1
            eps_length += 1
            s = ns
            a = self.policy.pi(s, terminal, possible_actions)

        return S, A, NS, R, T
