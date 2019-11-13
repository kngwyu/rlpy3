"""Gridworld based deepsea implementation.
The original version was introduced in https://arxiv.org/abs/1703.07608.
"""
import numpy as np
from .grid_world import GridWorld

__author__ = "Yuji Kanagawa"


class DeepSea(GridWorld):
    #: Down, Left
    ACTIONS = np.array([[0, -1], [0, +1]])
    ARROW_NAMES = ["LEFT", "RIGHT"]
    PIT_REWARD = 0.0
    STEP_REWARD = 0.0
    COLOR_MAP = "DeepSea"

    def __init__(self, size=10, randomize=True):
        map_ = np.zeros((size + 1, size + 1), dtype=np.int64)
        map_[0, 0] = self.START
        map_[-1, :] = self.PIT
        map_[:, -1] = self.GOAL
        # Based on the code in https://sites.google.com/view/randomized-prior-nips-2018
        map_[-1, -2] = self.GOAL
        self.size = size
        self._init_from_map(map_, "DeepSea-{}".format(size), False, 0.0, size + 1)
        if randomize:
            self._action_mapping = self.random_state.binomial(1, 0.5, size)
        else:
            self._action_mapping = np.ones(size, dtype=np.int64)
        self._move_cost = 0.01 / size

    def _agent_fig(self, s):
        fig, *_ = self.domain_ax.plot(
            s[1], s[0], "k>", markersize=20 - self.cols, color="xkcd:lemon"
        )
        return fig

    def step(self, a):
        row, col = self.state
        action_right = int(a) == self._action_mapping[col]

        reward = 0.0
        # New state
        if action_right:
            new_col = min(col + 1, self.size - 1)
            reward -= self._move_cost
        else:
            new_col = max(col - 1, 0)
        ns = np.array([row + 1, new_col])
        self.state = ns.copy()
        # Termination
        terminal = self.is_terminal()
        if self.map[int(ns[0]), int(ns[1])] == self.GOAL:
            reward += self.GOAL_REWARD
        return reward, ns, terminal, self.possible_actions()

    def possible_actions(self, s=None):
        return np.arange(2)

    def expected_step(self, s, a):
        raise NotImplementedError()
