"""Simple Chain MDP domain."""
from rlpy.tools import plt, mpatches, fromAtoB
import numpy as np
from .domain import Domain

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


class ChainMDP(Domain):

    """
    A simple Chain MDP.

    **STATE:** s0 <-> s1 <-> ... <-> sn \n
    **ACTIONS:** are left [0] and right [1], deterministic. \n

    .. note::

        The actions [left, right] are available in ALL states, but if
        left is selected in s0 or right in sn, then s remains unchanged.

    The task is to reach sn from s0, after which the episode terminates.

    .. note::
        Optimal policy is to always to go right.

    **REWARD:**
    -1 per step, 0 at goal (terminates)

    **REFERENCE:**

    .. seealso::
        Michail G. Lagoudakis, Ronald Parr, and L. Bartlett
        Least-squares policy iteration.  Journal of Machine Learning Research
        (2003) Issue 4.
    """

    #: Reward for each timestep spent in the goal region
    GOAL_REWARD = 0
    #: Reward for each timestep
    STEP_REWARD = -1
    # Used for graphical normalization
    MAX_RETURN = 1
    # Used for graphical normalization
    MIN_RETURN = 0
    # Used for graphical shifting of arrows
    SHIFT = 0.3
    # Used for graphical radius of states
    RADIUS = 0.5
    # Y values used for drawing circles
    Y = 1

    def __init__(self, chain_size=2):
        """
        :param chain_size: Number of states \'n\' in the chain.
        """
        self.chain_size = chain_size
        self.start = 0
        self.goal = chain_size - 1
        # Stores the graphical pathes for states so that we can later change their
        # colors
        self.circles = None
        self.fig = None
        super().__init__(
            actions_num=2,
            statespace_limits=np.array([[0, chain_size - 1]]),
            episode_cap=2 * chain_size,
        )

    def show_domain(self, a=0):
        # Draw the environment
        if self.circles is None:
            self.fig = plt.figure(1, (self.chain_size * 2, 2))
            ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.0)
            ax.set_xlim(0, self.chain_size * 2)
            ax.set_ylim(0, 2)
            # Make the last one double circle
            ax.add_patch(
                mpatches.Circle(
                    (1 + 2 * (self.chain_size - 1), self.Y), self.RADIUS * 1.1, fc="w"
                )
            )
            self.circles = [
                mpatches.Circle((1 + 2 * i, self.Y), self.RADIUS, fc="w")
                for i in range(self.chain_size)
            ]
            for i in range(self.chain_size):
                ax.add_patch(self.circles[i])
                if i < self.chain_size - 1:
                    fromAtoB(
                        1 + 2 * i + self.SHIFT,
                        self.Y + self.SHIFT,
                        1 + 2 * (i + 1) - self.SHIFT,
                        self.Y + self.SHIFT,
                    )
                if i < self.chain_size - 2:
                    fromAtoB(
                        1 + 2 * (i + 1) - self.SHIFT,
                        self.Y - self.SHIFT,
                        1 + 2 * i + self.SHIFT,
                        self.Y - self.SHIFT,
                        "r",
                    )
                fromAtoB(
                    0.75,
                    self.Y - 1.5 * self.SHIFT,
                    0.75,
                    self.Y + 1.5 * self.SHIFT,
                    "r",
                    connectionstyle="arc3,rad=-1.2",
                )
            self.fig.show()

        for i, p in enumerate(self.circles):
            if self.state[0] == i:
                p.set_facecolor("k")
            else:
                p.set_facecolor("w")
        self.fig.canvas.draw()

    def step(self, a):
        s = self.state[0]
        if a == 0:  # left
            ns = max(0, s - 1)
        if a == 1:
            ns = min(self.chain_size - 1, s + 1)
        ns = np.array([ns])
        self.state = ns

        terminal = self.is_terminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal, self.possible_actions()

    def s0(self):
        self.state = np.array([0])
        return self.state, self.is_terminal(), self.possible_actions()

    def is_terminal(self):
        s = self.state
        return s[0] == self.chain_size - 1
