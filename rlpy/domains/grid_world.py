"""Gridworld Domain."""
import numpy as np
import itertools
from rlpy.tools import FONTSIZE, linear_map, plt
from rlpy.tools import __rlpy_location__, findElemArray1D, perms
import os

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


class GridWorld(Domain):
    """
    The GridWorld domain simulates a path-planning problem for a mobile robot
    in an environment with obstacles. The goal of the agent is to
    navigate from the starting point to the goal state.
    The map is loaded from a text file filled with numbers showing the map with the
    following coding for each cell:

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
    MIN_RETURN = -1
    # Used for graphical shifting of arrows
    SHIFT = 0.1
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = range(6)
    #: Up, Down, Left, Right
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
    # directory of maps shipped with rlpy
    DEFAULT_MAP_DIR = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    # Keys to access arrow figures
    ARROW_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    # Color map to visualize the grid
    COLOR_MAP = "GridWorld"

    @classmethod
    def default_map(cls, name="4x5.txt"):
        return os.path.join(cls.DEFAULT_MAP_DIR, name)

    @staticmethod
    def _load_map(mapfile):
        map_ = np.loadtxt(mapfile, dtype=np.uint8)
        if map_.ndim == 1:
            return np.expand_dims(map_, 0)
        else:
            return map_

    def __init__(
        self,
        mapfile=os.path.join(DEFAULT_MAP_DIR, "4x5.txt"),
        noise=0.1,
        random_start=False,
        episode_cap=1000,
    ):
        map_ = self._load_map(mapfile)
        mapfname = os.path.basename(mapfile)
        dot_pos = mapfname.find(".")
        if dot_pos == -1:
            mapname = mapfname
        else:
            mapname = mapfname[:dot_pos]

        self._init_from_map(map_, mapname, random_start, noise, episode_cap)

    def _init_from_map(self, map_, mapname, random_start, noise, episode_cap):
        self.map = map_
        self.random_start = random_start
        # Number of rows and columns of the map
        self.rows, self.cols = self.map.shape
        super().__init__(
            actions_num=4,
            statespace_limits=np.array([[0, self.rows - 1], [0, self.cols - 1]]),
            episode_cap=episode_cap,
        )
        # Movement noise
        self.noise = noise
        self.DimNames = ["Row", "Col"]
        self.state = self._sample_start()
        # map name for the viewer title
        self.mapname = mapname
        # Used for graphics to show the domain
        self.domain_fig, self.domain_ax, self.agent_fig = None, None, None
        self.vf_fig, self.vf_ax, self.vf_img = None, None, None
        self.arrow_figs = {}
        self.goal_reward = self.MAX_RETURN
        self.pit_reward = self.MIN_RETURN
        self.vf_texts = []
        self.r_fig, self.r_ax, self.r_img = None, None, None
        self.r_texts = []

    def _sample_start(self):
        starts = np.argwhere(self.map == self.START)
        if self.random_start:
            idx = self.random_state.randint(len(starts))
        else:
            idx = 0
        self.start_state = starts[idx]
        return self.start_state.copy()

    def _show_map(self):
        cmap = plt.get_cmap(self.COLOR_MAP)
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
        ax.set_xticks(np.arange(self.cols))
        xlabels = ax.get_xticklabels()
        for l in xlabels:
            l.update({"fontsize": FONTSIZE})

        ax.set_yticks(np.arange(self.rows))
        ylabels = ax.get_yticklabels()
        for l in ylabels:
            l.update({"fontsize": FONTSIZE})

    def _agent_fig(self, s):
        return self.domain_ax.plot(s[1], s[0], "k>", markersize=20 - self.cols)[0]

    def _init_domain_vis(self, s):
        fig_name = "GridWorld: {}".format(self.mapname)
        if self.performance:
            fig_name += "(Evaluation)"
        self.domain_fig = plt.figure(fig_name)
        ratio = self.rows / self.cols
        self.domain_ax = self.domain_fig.add_axes((0.08, 0.04, 0.86 * ratio, 0.86))
        self._show_map()
        self._set_ticks(self.domain_ax)
        self.agent_fig = self._agent_fig(s)
        self.domain_fig.show()

    def show_domain(self, a=0, s=None):
        if s is None:
            s = self.state
        # Draw the environment
        if self.domain_fig is None:
            self._init_domain_vis(s)
        self.agent_fig.remove()
        self.agent_fig = self._agent_fig(s)
        self.domain_fig.canvas.draw()

    def _init_vis_common(self, fig):
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap("ValueFunction-New")
        img = ax.imshow(
            self.map,
            cmap=cmap,
            interpolation="nearest",
            vmin=self.MIN_RETURN,
            vmax=self.MAX_RETURN,
        )
        ax.plot([0.0], [0.0], color=cmap(256), label="Max")
        ax.plot([0.0], [0.0], color=cmap(0), label="Min")
        ax.legend(fontsize=12, bbox_to_anchor=(1, 1))
        self._set_ticks(ax)
        return ax, img

    def _init_reward_vis(self, r):
        self.r_fig = plt.figure("Pseudo Reward")
        self.r_ax, self.r_img = self._init_vis_common(self.r_fig)
        self.r_ax.plot(0, 0)
        self.r_fig.show()

    def show_reward(self, reward_):
        """
        Visualize learned reward functions for PSRL or other methods.
        """
        reward = reward_.reshape(self.cols, self.rows).T
        if self.r_fig is None:
            self._init_reward_vis(reward)

        for txt in self.r_texts:
            txt.remove()
        self.r_texts.clear()
        rmin, rmax = reward.min(), reward.max()
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if reward[r, c] == rmin:
                self._vf_text(c, r, rmin, mode="r")
            elif reward[r, c] == rmax:
                self._vf_text(c, r, rmax, mode="r")
            if reward[r, c] < 0:
                reward[r, c] = linear_map(
                    reward[r, c], min(rmin, self.MIN_RETURN), 0, -1, 0
                )
            else:
                reward[r, c] = linear_map(
                    reward[r, c], 0, max(rmax, self.MAX_RETURN), 0, 1
                )
        self.r_img.set_data(reward)
        self.r_fig.canvas.draw()

    def _vf_text(self, c, r, v, mode="vf"):
        if mode == "vf":
            cache = self.vf_texts
            ax = self.vf_ax
        else:
            cache = self.r_texts
            ax = self.r_ax
        cache.append(
            ax.text(c - 0.2, r + 0.1, format(v, ".1f"), color="xkcd:bright blue")
        )

    def _init_arrow(self, name, x, y):
        arrow_ratio = 0.4
        MAX_RATIO_HEAD_TO_LENGTH = 0.25
        ARROW_WIDTH = 0.5 * MAX_RATIO_HEAD_TO_LENGTH / 5.0
        is_y = name in ["UP", "DOWN"]
        c = np.zeros(x.shape)
        c[0, 0] = 1
        self.arrow_figs[name] = self.vf_ax.quiver(
            x,
            y,
            np.ones(x.shape),
            np.ones(y.shape),
            c,
            units="y" if is_y else "x",
            cmap="Actions",
            scale_units="height" if is_y else "width",
            scale=(self.rows if is_y else self.cols) / arrow_ratio,
            width=-ARROW_WIDTH if is_y else ARROW_WIDTH,
        )
        self.arrow_figs[name].set_clim(vmin=0, vmax=1)

    def _init_value_vis(self):
        self.vf_fig = plt.figure("Value Function")
        self.vf_ax, self.vf_img = self._init_vis_common(self.vf_fig)
        # Create quivers for each action. 4 in total
        shift = self.ACTIONS * self.SHIFT
        x, y = np.arange(self.cols), np.arange(self.rows)
        for name, s in zip(self.ARROW_NAMES, shift):
            self._init_arrow(name, *np.meshgrid(x + s[1], y + s[0]))
        self.vf_fig.show()

    def show_learning(self, representation):
        if self.vf_ax is None:
            self._init_value_vis()
        for txt in self.vf_texts:
            txt.remove()
        self.vf_texts.clear()
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        arrow_mask = np.ones((self.rows, self.cols, self.actions_num), dtype="bool")
        arrow_size = np.zeros(arrow_mask.shape, dtype="float")
        # 0 = suboptimal action, 1 = optimal action
        arrow_color = np.zeros(arrow_mask.shape, dtype="uint8")
        v = np.zeros((self.rows, self.cols))
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            cell = self.map[r, c]
            if cell == self.BLOCKED:
                v[r, c] = 0
            elif cell in (self.START, self.EMPTY):
                s = np.array([r, c])
                actions = self.possible_actions(s)
                terminal = self.is_terminal(s)
                q_values = representation.Qs(s, terminal)
                best_act = representation.best_actions(s, terminal, actions)
                v[r, c] = q_values[actions].max()
                arrow_mask[r, c, actions] = False
                arrow_color[r, c, best_act] = 1
                for a, Q in zip(actions, q_values):
                    arrow_size[r, c, a] = linear_map(
                        Q, self.MIN_RETURN, self.MAX_RETURN
                    )

        vmin, vmax = v.min(), v.max()
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if v[r, c] == vmin:
                self._vf_text(c, r, vmin)
            elif v[r, c] == vmax:
                self._vf_text(c, r, vmax)
            if v[r, c] < 0:
                v[r, c] = linear_map(v[r, c], min(vmin, self.MIN_RETURN), 0, -1, 0)
            else:
                v[r, c] = linear_map(v[r, c], 0, max(vmax, self.MAX_RETURN), 0, 1)

        # Show Value Function
        self.vf_img.set_data(v)
        # Show Policy for arrows
        for i, name in enumerate(self.ARROW_NAMES):
            dy, dx = self.ACTIONS[i]
            size, mask = arrow_size[:, :, i], arrow_mask[:, :, i]
            dx = np.ma.masked_array(dx * size, mask=mask)
            dy = np.ma.masked_array(dy * size * -1, mask=mask)
            c = np.ma.masked_array(arrow_color[:, :, i], mask=mask)
            self.arrow_figs[name].set_UVC(dx, dy, c)
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
            a = self.random_state.choice(self.possible_actions())

        # Take action
        ns = self.state + self.ACTIONS[a]

        # Check bounds on state values
        if self._valid_state(ns) and self.map[ns[0], ns[1]] != self.BLOCKED:
            # If in bounds, update the current state
            self.state = ns.copy()
        else:
            ns = self.state.copy()

        terminal = self.is_terminal()
        reward = self._reward(ns, terminal)
        return reward, ns, terminal, self.possible_actions()

    def s0(self):
        self.state = self._sample_start()
        return self.state, self.is_terminal(), self.possible_actions()

    def _valid_state(self, state):
        y, x = state
        return 0 <= y < self.rows and 0 <= x < self.cols

    def is_terminal(self, s=None):
        if s is None:
            s = self.state
        cell = self.map[int(s[0]), int(s[1])]
        return cell == self.GOAL or cell == self.PIT

    def possible_actions(self, s=None):
        if s is None:
            s = self.state
        s = s.astype(np.int64)
        possible_a = []
        for a in range(self.actions_num):
            ns = s + self.ACTIONS[a]
            if self._valid_state(ns) and self.map[ns[0], ns[1]] != self.BLOCKED:
                possible_a.append(a)
        return np.array(possible_a)

    def expected_step(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        # pa: k-by-??   possible actions for each next state
        actions = self.possible_actions(s)
        k = len(actions)
        # Make Probabilities
        intended_action_index = findElemArray1D(a, actions)
        p = np.ones((k, 1)) * self.noise / (k * 1.0)
        p[intended_action_index, 0] += 1 - self.noise
        # Make next states
        ns = np.tile(s, (k, 1)).astype(int) + self.ACTIONS[actions]
        # Make next possible actions
        pa = np.array([self.possible_actions(sn) for sn in ns])
        # Make rewards
        r = np.ones((k, 1)) * self.STEP_REWARD
        goal = self.map[ns[:, 0], ns[:, 1]] == self.GOAL
        pit = self.map[ns[:, 0], ns[:, 1]] == self.PIT
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
                perms(self.discrete_statespace_width + 1)
                + self.discrete_statespace_limits[0]
            )
        else:
            return None
