"""Gridworld Domain."""
from collections import defaultdict
import numpy as np
import itertools
from rlpy.tools import FONTSIZE, linear_map, plt, with_scaled_figure
from rlpy.tools import __rlpy_location__, findElemArray1D
from pathlib import Path

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
    DEFAULT_MAP_DIR = Path(__rlpy_location__).joinpath("domains/GridWorldMaps")
    # Keys to access arrow figures
    ARROW_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
    # Color map to visualize the grid
    COLOR_MAP = "GridWorld"

    @classmethod
    def default_map(cls, name="4x5.txt"):
        return cls.DEFAULT_MAP_DIR.joinpath(name)

    @staticmethod
    def _load_map(mapfile):
        map_ = np.loadtxt(mapfile, dtype=np.uint8)
        if map_.ndim == 1:
            return np.expand_dims(map_, 0)
        else:
            return map_

    def __init__(
        self,
        mapfile=DEFAULT_MAP_DIR.joinpath("4x5.txt"),
        noise=0.1,
        random_start=False,
        random_goal=False,
        episode_cap=1000,
    ):
        if isinstance(mapfile, str):
            mapfile = Path(mapfile)
        map_ = self._load_map(mapfile)
        mapname = mapfile.stem

        self._init_from_map(
            map_, mapname, noise, episode_cap, random_start, random_goal
        )

    def _init_from_map(
        self, map_, mapname, noise, episode_cap, random_start=False, random_goal=False
    ):
        self.map = map_
        self.random_start = random_start
        self._goals = np.argwhere(map_ == self.GOAL)
        self.random_goal = random_goal
        # Number of rows and columns of the map
        self.rows, self.cols = self.map.shape
        super().__init__(
            actions_num=4,
            statespace_limits=np.array([[0, self.rows - 1], [0, self.cols - 1]]),
            episode_cap=episode_cap,
        )
        # Movement noise
        self.noise = noise
        self.dim_names = ["Row", "Col"]
        self.state = self._sample_start()
        # map name for the viewer title
        self.mapname = mapname
        # Used for graphics to show the domain
        self.domain_fig, self.domain_ax, self.domain_img, self.agent_fig = (None,) * 4
        self.goal_changed = True
        self.vf_fig, self.vf_ax, self.vf_img = (None,) * 3
        self.arrow_figs = []
        self.goal_reward = self.MAX_RETURN
        self.pit_reward = self.MIN_RETURN
        self.vf_texts = []
        self.heatmap_fig, self.heatmap_ax, self.heatmap_img = {}, {}, {}
        self.heatmap_texts = defaultdict(list)
        self.policy_fig, self.policy_ax, self.policy_img = None, {}, {}
        self.policy_arrows, self.policy_texts = defaultdict(list), defaultdict(list)

    def _sample_start(self):
        starts = np.argwhere(self.map == self.START)
        if self.random_start:
            idx = self.random_state.randint(starts.shape[0])
        else:
            idx = 0
        self.start_state = starts[idx]
        return self.start_state.copy()

    def _sample_goal(self):
        idx = self.random_state.randint(self._goals.shape[0])
        for i, (r, c) in enumerate(self._goals):
            if i == idx:
                self.map[r, c] = self.GOAL
            else:
                self.map[r, c] = self.EMPTY
        self.goal_changed = True

    def _map_mask(self):
        return (self.map != self.BLOCKED).astype(np.float32)

    def _legend_pos(self):
        x_offset, y_offset = 1.1, 1.1
        if self.rows > self.cols:
            x_offset += min(1.0, 0.25 * self.rows / self.cols)
        if self.cols > self.rows:
            y_offset += min(1.0, 0.25 * self.cols / self.rows)
        return x_offset, y_offset

    def _show_map(self):
        cmap = plt.get_cmap(self.COLOR_MAP)
        self.domain_img = self.domain_ax.imshow(
            self.map, cmap=cmap, interpolation="nearest", vmin=0, vmax=5
        )
        self.domain_ax.plot([0.0], [0.0], color=cmap(1), label="Block")
        self.domain_ax.plot([0.0], [0.0], color=cmap(2), label="Start")
        self.domain_ax.plot([0.0], [0.0], color=cmap(3), label="Goal")
        self.domain_ax.plot([0.0], [0.0], color=cmap(4), label="Pit")
        self.domain_ax.legend(
            fontsize=12, loc="upper right", bbox_to_anchor=self._legend_pos()
        )

    def _set_ticks(self, ax, fontsize=FONTSIZE):
        ax.get_xaxis().set_ticks_position("top")
        ax.set_xticks(np.arange(self.cols))
        xlabels = ax.get_xticklabels()
        for l in xlabels:
            l.update({"fontsize": FONTSIZE})

        ax.set_yticks(np.arange(self.rows))
        ylabels = ax.get_yticklabels()
        for l in ylabels:
            l.update({"fontsize": FONTSIZE})

    def _noticks(self, ax, fontsize=FONTSIZE):
        ax.set_xticks([])
        ax.set_yticks([])

    def _agent_fig(self, s):
        return self.domain_ax.plot(s[1], s[0], "k>", markersize=20 - self.cols)[0]

    def _init_domain_vis(self, s):
        fig_name = "GridWorld: {}".format(self.mapname)
        if self.performance:
            fig_name += "(Evaluation)"
        self.domain_fig = plt.figure(fig_name)
        self.domain_ax = self.domain_fig.add_subplot(111)
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
        if self.goal_changed:
            self.domain_img.set_data(self.map)
            self.goal_changed = False
        self.agent_fig.remove()
        self.agent_fig = self._agent_fig(s)
        self.domain_fig.canvas.draw()

    def _init_vis_common(
        self,
        fig,
        cmap="ValueFunction-New",
        axarg=(1, 1, 1),
        legend=True,
        ticks=True,
        cmap_vmin=MIN_RETURN,
        cmap_vmax=MAX_RETURN,
    ):
        ax = fig.add_subplot(*axarg)
        cmap = plt.get_cmap(cmap)
        img = ax.imshow(
            self.map,
            cmap=cmap,
            interpolation="nearest",
            vmin=cmap_vmin,
            vmax=cmap_vmax,
        )
        ax.plot([0.0], [0.0], color=cmap(256), label=f"Max")
        ax.plot([0.0], [0.0], color=cmap(0), label=f"Min")
        if legend:
            ax.legend(fontsize=12, bbox_to_anchor=self._legend_pos())
        if ticks:
            self._set_ticks(ax)
        else:
            self._noticks(ax)
        return ax, img

    def _init_heatmap_vis(
        self, name, cmap, nrows, ncols, index, legend, ticks, cmap_vmin, cmap_vmax
    ):
        if name not in self.heatmap_fig:
            self.heatmap_fig[name] = plt.figure(name)
            self.heatmap_fig[name].show()

        ax, img = self._init_vis_common(
            self.heatmap_fig[name],
            cmap=cmap,
            axarg=(nrows, ncols, index),
            legend=legend,
            ticks=ticks,
            cmap_vmin=cmap_vmin,
            cmap_vmax=cmap_vmax,
        )
        self.heatmap_ax[name, index], self.heatmap_img[name, index] = ax, img

    def show_reward(self, reward_):
        """
        Visualize learned reward functions for PSRL or other methods.
        """
        reward = reward_.reshape(self.cols, self.rows).T
        self.show_heatmap(reward, "Pseudo Reward")

    def _normalize_separated(self, value, vmin, vmax, cmap_vmin, cmap_vmax):
        if value < 0:
            return linear_map(value, min(vmin, cmap_vmin), 0, cmap_vmin, 0)
        else:
            return linear_map(value, 0, max(vmax, cmap_vmax), 0, cmap_vmax)

    def _normalize_uniform(self, value, vmin, vmax, cmap_vmin, cmap_vmax):
        vmin = min(vmin, cmap_vmin)
        vmax = max(vmax, cmap_vmax)
        return

    def _normalize_value(
        self,
        value,
        method="separated",
        scaling=True,
        cmap_vmin=MIN_RETURN,
        cmap_vmax=MAX_RETURN,
    ):

        vmin, vmax = value.min(), value.max()
        vmin_coord, vmax_coord = None, None

        if scaling:
            vmin_scaled = min(vmin, cmap_vmin)
            vmax_scaled = min(vmax, cmap_vmax)
        else:
            vmin_scaled = vmin
            vmax_scaled = vmax

        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if value[r, c] == vmin and vmin_coord is None:
                vmin_coord = r, c, vmin
            elif value[r, c] == vmax and vmax_coord is None:
                vmax_coord = r, c, vmax
            if method == "separated":
                if value[r, c] < 0:
                    value[r, c] = linear_map(value[r, c], vmin_scaled, 0, cmap_vmin, 0)
                else:
                    value[r, c] = linear_map(value[r, c], 0, vmax_scaled, 0, cmap_vmax)
            elif method == "uniform":
                value[r, c] = linear_map(
                    value[r, c], vmin_scaled, vmax_scaled, cmap_vmin, cmap_vmax
                )

        return vmin_coord, vmax_coord

    def show_heatmap(
        self,
        value,
        name,
        normalize_method="separated",
        cmap="ValueFunction-New",
        title=None,
        nrows=1,
        ncols=1,
        index=1,
        legend=True,
        ticks=True,
        scaling=True,
        cmap_vmin=MIN_RETURN,
        cmap_vmax=MAX_RETURN,
    ):
        """
        Visualize learned reward functions for PSRL or other methods.
        """
        if len(value.shape) == 1:
            value = value.reshape(self.rows, self.cols)

        key = name, index

        if key not in self.heatmap_ax:
            self._init_heatmap_vis(
                name, cmap, nrows, ncols, index, legend, ticks, cmap_vmin, cmap_vmax
            )
            if title is not None:
                self.heatmap_ax[key].set_title(title)

        self._reset_texts(self.heatmap_texts[key])

        coords = self._normalize_value(
            value,
            method=normalize_method,
            scaling=scaling,
            cmap_vmin=cmap_vmin,
            cmap_vmax=cmap_vmax,
        )

        for r, c, ext_v in coords:
            self._text_on_cell(
                c, r, ext_v, self.heatmap_texts[key], self.heatmap_ax[key]
            )

        self.heatmap_img[key].set_data(value * self._map_mask())
        self.heatmap_fig[name].canvas.draw()

    def _vf_text(self, c, r, v):
        self._text_on_cell(c, r, v, self.vf_texts, self.vf_ax)

    @staticmethod
    def _reset_texts(texts):
        for txt in texts:
            txt.remove()
        texts.clear()

    @staticmethod
    def _text_on_cell(c, r, v, cache, ax):
        cache.append(
            ax.text(c - 0.2, r + 0.1, format(v, ".1f"), color="xkcd:bright blue")
        )

    def _init_arrow(self, name, x, y, ax, arrow_scale=1.0):
        arrow_ratio = 0.4 * arrow_scale
        ARROW_WIDTH = 0.05
        is_y = name in ["UP", "DOWN"]
        c = np.zeros(x.shape)
        c[0, 0] = 1
        arrow_fig = ax.quiver(
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
        arrow_fig.set_clim(vmin=0, vmax=1)
        return arrow_fig

    def show_policy(
        self,
        policy,
        value=None,
        nrows=1,
        ncols=1,
        index=1,
        ticks=True,
        scale=1.0,
        title=None,
    ):
        if self.policy_fig is None:
            with with_scaled_figure(scale):
                self.policy_fig = plt.figure("Policy")
            self.policy_fig.show()
        if index not in self.policy_ax:
            self.policy_ax[index], self.policy_img[index] = self._init_vis_common(
                self.policy_fig, axarg=(nrows, ncols, index), legend=False, ticks=ticks
            )
            shift = self.ACTIONS * self.SHIFT
            x, y = np.arange(self.cols), np.arange(self.rows)
            for name, s in zip(self.ARROW_NAMES, shift):
                grid = np.meshgrid(x + s[1], y + s[0])
                self.policy_arrows[index].append(
                    self._init_arrow(
                        name, *grid, self.policy_ax[index], arrow_scale=scale,
                    )
                )

            if title is not None:
                self.policy_ax[index].set_title(title)

        arrow_mask = np.ones((self.rows, self.cols, self.actions_num), dtype=np.bool)
        arrow_size = np.zeros(arrow_mask.shape, dtype=np.float32)
        arrow_color = np.zeros(arrow_mask.shape, dtype=np.uint8)

        try:
            policy = policy.reshape(self.rows, self.cols, -1)
        except ValueError:
            raise ValueError(f"Invalid policy shape: {policy.shape}")
        _, _, action_dim = policy.shape

        for r, c in itertools.product(range(self.rows), range(self.cols)):
            cell = self.map[r, c]
            if cell not in (self.START, self.EMPTY):
                continue
            s = np.array([r, c])
            actions = self.possible_actions(s)
            best_act = policy[r, c].argmax()
            arrow_mask[r, c, actions] = False
            arrow_color[r, c, best_act] = 1
            for a in actions:
                arrow_size[r, c, a] = policy[r, c, a]

        # Show Policy for arrows
        for i, name in enumerate(self.ARROW_NAMES):
            dy, dx = self.ACTIONS[i]
            size, mask = arrow_size[:, :, i], arrow_mask[:, :, i]
            dx = np.ma.masked_array(dx * size, mask=mask)
            dy = np.ma.masked_array(dy * size * -1, mask=mask)
            c = np.ma.masked_array(arrow_color[:, :, i], mask=mask)
            self.policy_arrows[index][i].set_UVC(dx, dy, c)

        if value is not None:
            try:
                value = value.reshape(self.rows, self.cols)
            except ValueError:
                raise ValueError(f"Invalid value shape: {value.shape}")

            self._reset_texts(self.policy_texts[index])
            for r, c, ext_v in self._normalize_value(value):
                self._text_on_cell(
                    c, r, ext_v, self.policy_texts[index], self.policy_ax[index]
                )
            self.policy_img[index].set_data(value * self._map_mask())
        self.policy_fig.canvas.draw()

    def _init_value_vis(self):
        self.vf_fig = plt.figure("Value Function")
        self.vf_ax, self.vf_img = self._init_vis_common(self.vf_fig)
        # Create quivers for each action. 4 in total
        shift = self.ACTIONS * self.SHIFT
        x, y = np.arange(self.cols), np.arange(self.rows)
        for name, s in zip(self.ARROW_NAMES, shift):
            self.arrow_figs.append(
                self._init_arrow(name, *np.meshgrid(x + s[1], y + s[0]), self.vf_ax)
            )
        self.vf_fig.show()

    def show_learning(self, representation):
        if self.vf_ax is None:
            self._init_value_vis()
        self._reset_texts(self.vf_texts)
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        arrow_mask = np.ones((self.rows, self.cols, self.actions_num), dtype=np.bool)
        arrow_size = np.zeros(arrow_mask.shape, dtype=np.float32)
        # 0 = suboptimal action, 1 = optimal action
        arrow_color = np.zeros(arrow_mask.shape, dtype=np.uint8)
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

        for r, c, ext_v in self._normalize_value(v):
            self._vf_text(c, r, ext_v)

        # Show Value Function
        self.vf_img.set_data(v)
        # Show Policy for arrows
        for i, name in enumerate(self.ARROW_NAMES):
            dy, dx = self.ACTIONS[i]
            size, mask = arrow_size[:, :, i], arrow_mask[:, :, i]
            dx = np.ma.masked_array(dx * size, mask=mask)
            dy = np.ma.masked_array(dy * size * -1, mask=mask)
            c = np.ma.masked_array(arrow_color[:, :, i], mask=mask)
            self.arrow_figs[i].set_UVC(dx, dy, c)
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
        if self.random_goal:
            self._sample_goal()
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

    def all_states(self):
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            yield np.array([r, c])

    def get_image(self, state):
        image = self.map.copy()
        image[state[0], state[1]] = self.AGENT
        return np.expand_dims(image.astype(np.float32), 0)
