"""Gridworld based deepsea implementation.
The original version was introduced in https://arxiv.org/abs/1703.07608.
"""
import numpy as np
from .grid_world import GridWorld

__author__ = "Yuji Kanagawa"


class DeepSea(GridWorld):
    #: Left Down / Right Down
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTIONS = np.array([[1, -1], [1, 1]])
    ALL_ACTIONS = np.array([[1, -1], [1, 1], [1, 0]])
    ARROW_NAMES = ["LEFT", "RIGHT"]
    PIT_REWARD = 0.0
    STEP_REWARD = 0.0
    COLOR_MAP = "DeepSea"

    def __init__(self, size=10, noise=0.0):
        map_ = np.zeros((size + 1, size + 1), dtype=np.int64)
        map_[0, 0] = self.START
        map_[-1, :] = self.PIT
        map_[:, -1] = self.GOAL
        self.size = size
        self._init_from_map(map_, "DeepSea-{}".format(size), False, noise, size + 1)
        self._move_cost = 0.01 / size

    def _agent_fig(self, s):
        fig, *_ = self.domain_ax.plot(
            s[1], s[0], "k>", markersize=20 - self.cols, color="xkcd:lemon"
        )
        return fig

    def is_terminal(self, s=None):
        if s is None:
            s = self.state
        return self.size <= max(s)

    def step(self, a):
        row, col = self.state.astype(int)
        action_right = int(a) == self.ACTION_RIGHT

        reward = 0.0
        # New state
        if action_right:
            if self.noise <= 0.0 or self.random_state.random_sample() >= self.noise:
                new_col = min(col + 1, self.size)
            else:
                new_col = col
            reward -= self._move_cost
        else:
            new_col = max(col - 1, 0)
        ns = np.array([row + 1, new_col])
        self.state = ns.copy()
        # Termination
        terminal = self.is_terminal(ns)
        if self.map[ns[0], ns[1]] == self.GOAL:
            reward += self.GOAL_REWARD

        return reward, ns, terminal, self.possible_actions()

    def possible_actions(self, s=None):
        return np.arange(2)

    def expected_step(self, s, a):
        s = s.astype(int)
        current_action = int(a)
        k = len(self.ALL_ACTIONS)
        # Make Probabilities
        # 0: Left 1: Right 2: NoOP
        p = np.zeros((k, 1))
        if current_action == self.ACTION_LEFT:
            p[current_action, :] = 1.0
        else:
            p[current_action, :] = 1.0 - self.noise
            p[2, :] = self.noise
        # Make next states
        all_actions = self.ALL_ACTIONS[np.arange(k)]
        ns = np.clip(np.tile(s, (k, 1)) + all_actions, 0, self.size)
        # Make next possible actions
        pa = np.tile(np.arange(k), (k, 1))
        # Make rewards
        r = np.zeros((k, 1))
        goal = self.map[ns[:, 0], ns[:, 1]] == self.GOAL
        r[goal] += self.GOAL_REWARD
        r[1, :] -= self._move_cost
        # Make terminals
        t = np.zeros((k, 1), bool)
        t[goal] = True
        pit = self.map[ns[:, 0], ns[:, 1]] == self.PIT
        t[pit] = True
        return p, r, ns, t, pa
