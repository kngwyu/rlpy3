"""Standard Experiment for Learning Control in RL"""
from .experiment import Experiment

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"


class MDPSolverExperiment(Experiment):
    """
    The MDPSolver Experiment connects an MDPSolver and Domain, and runs the MDPSolver's
    solve method to start solving the MDP.
    """

    def __init__(self, agent, domain, seed=10):
        """
        :param agent: The agent to be tested.
        :param domain: The domain to be tested on.
        """
        self.agent = agent
        self.domain = domain
        self.seed = seed

    def run(self, visualize=False):
        """
        Run the experiment and collect statistics / generate the results
        """
        self.domain.set_seed(self.seed)

        self.agent.solve(visualize=visualize)

    def performance_run(self, visualize=False):
        ret, _, _, _ = self.agent.performance_run(visualize=visualize)
        return ret
