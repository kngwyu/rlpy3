"""Standard Experiment for Learning Control in RL"""
import rlpy.tools.ipshell
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

    def __init__(self, agent, domain, seed=1):
        """
        :param agent: The agent to be tested.
        :param domain: The domain to be tested on.
        """
        self.agent = agent
        self.domain = domain
        self.seed = seed

    def run(self, visualize=False, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results

        debug_on_sigurg (boolean):
            if true, the ipdb debugger is opened when the python process
            receives a SIGURG signal. This allows to enter a debugger at any
            time, e.g. to view data interactively or actual debugging.
            The feature works only in Unix systems. The signal can be sent
            with the kill command:

                kill -URG pid

            where pid is the process id of the python interpreter running this
            function.

        """
        self.domain.set_seed(self.seed)
        if debug_on_sigurg:
            rlpy.tools.ipshell.ipdb_on_SIGURG()

        self.agent.solve(visualize=visualize)

    def performance_run(self, visualize=False):
        ret, _, _, _ = self.agent.performance_run(visualize=visualize)
        return ret
