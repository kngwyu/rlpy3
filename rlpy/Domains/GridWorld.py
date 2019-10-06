"""Gridworld Domain."""
import numpy as np
import itertools
from rlpy.Tools import plt, FONTSIZE, linearMap
from rlpy.Tools import __rlpy_location__, findElemArray1D, perms
import os

from .Domain import Domain

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


class GridWorld(Domain):
    """
    The GridWorld domain simulates a path-planning problem for a mobile robot
    in an environment with obstacles. The goal of the agent is to
    navigate from the starting point to the goal state.
    The map is loaded from a text file filled with numbers showing the map with the following
    coding for each cell:

    * 0: empty
    * 1: blocked
    * 2: start
    * 3: goal
    * 4: pit

    **STATE:**
    The Row and Column corresponding to the agent's location. \n
    **ACTIONS:**
    Four cardinal directions: up, down, left, right (given that
    the destination is not blocked or out of range). \n
    **TRANSITION:**
    There is 30% probability of failure for each move, in which case the action
    is replaced with a random action at each timestep. Otherwise the move succeeds
    and the agent moves in the intended direction. \n
    **REWARD:**
    The reward on each step is -.001 , except for actions
    that bring the agent to the goal with reward of +1.\n

    """

    #: Reward constants
    GOAL_REWARD = +1
    PIT_REWARD = -1
    STEP_REWARD = -0.001
    # Used for graphical normalization
    MAX_RETURN = 1
    RMAX = MAX_RETURN
    # Used for graphical normalization
    MIN_RETURN = -1
    # Used for graphical shifting of arrows
    SHIFT = 0.1
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = range(6)
    #: Up, Down, Left, Right
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
    # directory of maps shipped with rlpy
    DEFAULT_MAP_DIR = os.path.join(__rlpy_location__, "Domains", "GridWorldMaps")
    # Keys to access arrow figures
    ARROW_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

    @classmethod
    def default_map(cls, name="4x5.txt"):
        return os.path.join(cls.DEFAULT_MAP_DIR, name)

    def _load_map(self, mapfile):
        self.map = np.loadtxt(mapfile, dtype=np.uint8)
        if self.map.ndim == 1:
            self.map = self.map[np.newaxis, :]

    def __init__(
        self,
        mapfile=os.path.join(DEFAULT_MAP_DIR, "4x5.txt"),
        noise=0.1,
        random_start=False,
        episodeCap=1000,
    ):
        self._load_map(mapfile)
        self.random_start = random_start
        #: Number of rows and columns of the map
        self.rows, self.cols = np.shape(self.map)
        super().__init__(
            actions_num=4,
            statespace_limits=np.array([[0, self.rows - 1], [0, self.cols - 1]]),
            # 2*W*H, small values can cause problem for some planning techniques
            episodeCap=episodeCap,
        )
        #: Movement noise
        self.noise = noise
        self.DimNames = ["Row", "Col"]
        self.state = self._sample_start()
        # map name for showing
        mapfname = os.path.basename(mapfile)
        dot_pos = mapfname.find(".")
        if dot_pos == -1:
            self.mapname = mapfname
        else:
            self.mapname = mapfname[:dot_pos]
        # Used for graphics to show the domain
        self.domain_fig, self.domain_ax, self.agent_fig = None, None, None
        self.vf_fig, self.vf_ax, self.vf_img = None, None, None
        self.arrow_figs = {}

    def _sample_start(self):
        starts = np.argwhere(self.map == self.START)
        if self.random_start:
            idx = self.random_state.randint(len(starts))
        else:
            idx = 0
        self.start_state = starts[idx]
        return self.start_state.copy()

    def _show_map(self):
        cmap = plt.get_cmap("GridWorld")
        self.domain_ax.imshow(
            self.map, cmap=cmap, interpolation="nearest", vmin=0, vmax=5
        )
        self.domain_ax.plot([0.0], [0.0], color=cmap(1), label="Block")
        self.domain_ax.plot([0.0], [0.0], color=cmap(2), label="Start")
        self.domain_ax.plot([0.0], [0.0], color=cmap(3), label="Goal")
        self.domain_ax.plot([0.0], [0.0], color=cmap(4), label="Pit")
        self.domain_ax.legend(fontsize=12, loc="upper right", bbox_to_anchor=(1.2, 1.1))

    def _set_ticks(self, ax):
        ax.get_xaxis().set_ticks_position("top")
        plt.xticks(np.arange(self.cols), fontsize=FONTSIZE)
        plt.yticks(np.arange(self.rows), fontsize=FONTSIZE)

    def showDomain(self, a=0, s=None):
        if s is None:
            s = self.state
        # Draw the environment
        if self.domain_fig is None:
            self.domain_fig = plt.figure("GridWorld: {}".format(self.mapname))
            ratio = self.rows / self.cols
            self.domain_ax = self.domain_fig.add_axes((0.08, 0.04, 0.86 * ratio, 0.86))
            self._show_map()
            self._set_ticks(self.domain_ax)
            self.agent_fig = self.domain_ax.plot(
                s[1], s[0], "k>", markersize=20.0 - self.cols
            )[0]
            self.domain_fig.show()
        self.agent_fig.remove()
        self.agent_fig = self.domain_ax.plot(
            s[1], s[0], "k>", markersize=20.0 - self.cols
        )[0]
        self.domain_fig.canvas.draw()

    def _init_arrow(self, name, x, y):
        arrow_ratio = 0.4
        Max_Ratio_ArrowHead_to_ArrowLength = 0.25
        ARROW_WIDTH = 0.5 * Max_Ratio_ArrowHead_to_ArrowLength / 5.0
        is_y = name in ["UP", "DOWN"]
        c = np.zeros(x.shape)
        c[0, 0] = 1
        self.arrow_figs[name] = self.vf_ax.quiver(
            y,
            x,
            np.ones(x.shape),
            np.ones(x.shape),
            c,
            units="y" if is_y else "x",
            cmap="Actions",
            scale_units="height" if is_y else "width",
            scale=(self.rows if is_y else self.cols) / arrow_ratio,
            width=-ARROW_WIDTH if is_y else ARROW_WIDTH,
        )
        self.arrow_figs[name].set_clim(vmin=0, vmax=1)

    def showLearning(self, representation):
        if self.vf_ax is None:
            self.vf_fig = plt.figure("Value Function")
            self.vf_ax = self.vf_fig.add_subplot(1, 1, 1)
            cmap = plt.get_cmap("ValueFunction-New")
            self.vf_img = self.vf_ax.imshow(
                self.map,
                cmap=cmap,
                interpolation="nearest",
                vmin=self.MIN_RETURN,
                vmax=self.MAX_RETURN,
            )
            self.vf_ax.legend(fontsize=12, bbox_to_anchor=(1.3, 1.05))
            self._set_ticks(self.vf_ax)
            # Create quivers for each action. 4 in total
            xshift = [-self.SHIFT, self.SHIFT, 0, 0]
            yshift = [0, 0, -self.SHIFT, self.SHIFT]
            for name, xshift, yshift in zip(self.ARROW_NAMES, xshift, yshift):
                x = np.arange(self.rows) + xshift
                y = np.arange(self.cols) + yshift
                self._init_arrow(name, *np.meshgrid(x, y))
            self.vf_fig.show()
        V = np.zeros((self.rows, self.cols))
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        Mask = np.ones((self.cols, self.rows, self.actions_num), dtype="bool")
        arrowSize = np.zeros((self.cols, self.rows, self.actions_num), dtype="float")
        # 0 = suboptimal action, 1 = optimal action
        arrowColors = np.zeros((self.cols, self.rows, self.actions_num), dtype="uint8")
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if self.map[r, c] == self.BLOCKED:
                V[r, c] = 0
            elif self.map[r, c] == self.GOAL:
                V[r, c] = self.MAX_RETURN
            elif self.map[r, c] == self.PIT:
                V[r, c] = self.MIN_RETURN
            elif self.map[r, c] == self.EMPTY or self.map[r, c] == self.START:
                s = np.array([r, c])
                As = self.possibleActions(s)
                terminal = self.isTerminal(s)
                Qs = representation.Qs(s, terminal)
                bestA = representation.bestActions(s, terminal, As)
                V[r, c] = max(Qs[As])
                Mask[c, r, As] = False
                arrowColors[c, r, bestA] = 1
                for a, Q in zip(As, Qs):
                    value = linearMap(Q, self.MIN_RETURN, self.MAX_RETURN, 0, 1)
                    arrowSize[c, r, a] = value
        # Show Value Function
        self.vf_img.set_data(V)
        # Show Policy for arrows
        for i, name in enumerate(self.ARROW_NAMES):
            flip = -1 if name in ["DOWN", "LEFT"] else 1
            if name in ["UP", "DOWN"]:
                dx, dy = flip * arrowSize[:, :, i], np.zeros((self.rows, self.cols))
            else:
                dx, dy = np.zeros((self.rows, self.cols)), flip * arrowSize[:, :, i]
            dx = np.ma.masked_array(dx, mask=Mask[:, :, i])
            dy = np.ma.masked_array(dy, mask=Mask[:, :, i])
            c = np.ma.masked_array(arrowColors[:, :, i], mask=Mask[:, :, i])
            self.arrow_figs[name].set_UVC(dy, dx, c)
        self.vf_fig.canvas.draw()

    def _reward(self, next_state, _terminal):
        if self.map[next_state[0], next_state[1]] == self.GOAL:
            return self.GOAL_REWARD
        elif self.map[next_state[0], next_state[1]] == self.PIT:
            return self.PIT_REWARD
        else:
            return self.STEP_REWARD

    def step(self, a):
        ns = self.state.copy()
        if self.random_state.random_sample() < self.noise:
            # Random Move
            a = self.random_state.choice(self.possibleActions())

        # Take action
        ns = self.state + self.ACTIONS[a]

        # Check bounds on state values
        if (
            ns[0] < 0
            or ns[0] == self.rows
            or ns[1] < 0
            or ns[1] == self.cols
            or self.map[ns[0], ns[1]] == self.BLOCKED
        ):
            ns = self.state.copy()
        else:
            # If in bounds, update the current state
            self.state = ns.copy()

        terminal = self.isTerminal()
        reward = self._reward(ns, terminal)
        return reward, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self._sample_start()
        return self.state, self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        if self.map[int(s[0]), int(s[1])] == self.GOAL:
            return True
        if self.map[int(s[0]), int(s[1])] == self.PIT:
            return True
        return False

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)
        for a in range(self.actions_num):
            ns = s + self.ACTIONS[a]
            if (
                ns[0] < 0
                or ns[0] == self.rows
                or ns[1] < 0
                or ns[1] == self.cols
                or self.map[int(ns[0]), int(ns[1])] == self.BLOCKED
            ):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA

    def expectedStep(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        # pa: k-by-??   possible actions for each next state
        actions = self.possibleActions(s)
        k = len(actions)
        # Make Probabilities
        intended_action_index = findElemArray1D(a, actions)
        p = np.ones((k, 1)) * self.noise / (k * 1.0)
        p[intended_action_index, 0] += 1 - self.noise
        # Make next states
        ns = np.tile(s, (k, 1)).astype(int)
        actions = self.ACTIONS[actions]
        ns += actions
        # Make next possible actions
        pa = np.array([self.possibleActions(sn) for sn in ns])
        # Make rewards
        r = np.ones((k, 1)) * self.STEP_REWARD
        goal = self.map[ns[:, 0].astype(np.int), ns[:, 1].astype(np.int)] == self.GOAL
        pit = self.map[ns[:, 0].astype(np.int), ns[:, 1].astype(np.int)] == self.PIT
        r[goal] = self.GOAL_REWARD
        r[pit] = self.PIT_REWARD
        # Make terminals
        t = np.zeros((k, 1), bool)
        t[goal] = True
        t[pit] = True
        return p, r, ns, t, pa

    def allStates(self):
        if len(self.continuous_dims) > 0:
            # Recall that discrete dimensions are assumed to be integer
            return (
                perms(
                    self.discrete_statespace_limits[:, 1]
                    - self.discrete_statespace_limits[:, 0]
                    + 1
                )
                + self.discrete_statespace_limits[:, 0]
            )
        else:
            return None
