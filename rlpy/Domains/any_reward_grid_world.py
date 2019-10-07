"""AnyRewardGridworld Domain."""
import itertools
import numpy as np
from rlpy.Tools import __rlpy_location__, plt, with_bold_fonts
import os

from .GridWorld import GridWorld

__license__ = "BSD 3-Clause"
__author__ = "Yuji Kanagawa"


class AnyRewardGridWorld(GridWorld):
    """The same as GridWorld, but you can set any reward for each cell.
    """

    # directory of maps shipped with rlpy
    DEFAULT_MAP_DIR = os.path.join(
        __rlpy_location__, "Domains", "AnyRewardGridWorldMaps"
    )

    def _load_map(self, mapfile):
        map_and_reward = np.loadtxt(mapfile, dtype=np.int32)
        mshape = map_and_reward.shape
        if mshape[1] * 2 != mshape[0]:
            raise ValueError("Invalid map with shape {}".format(mshape))
        col = mshape[0] // 2
        self.map = map_and_reward[:col]
        self.reward_map = map_and_reward[col:]

    def __init__(
        self,
        mapfile=os.path.join(DEFAULT_MAP_DIR, "6x6guided.txt"),
        noise=0.1,
        step_penalty=1.0,
        random_start=False,
        episodeCap=20,
    ):
        super().__init__(
            mapfile=mapfile,
            noise=noise,
            random_start=random_start,
            episodeCap=episodeCap,
        )
        self.step_penalty = step_penalty

    def _reward(self, next_state, terminal):
        reward = self.reward_map[next_state[0], next_state[1]]
        if not terminal:
            reward -= self.step_penalty
        return reward

    def _rew_range(self):
        mi, ma = 1000, -1000
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if self.map[r, c] == self.EMPTY:
                mi = min(mi, self.reward_map[r, c])
                ma = max(ma, self.reward_map[r, c])
        if mi == 1000:
            mi = min(self.reward_map[r, c])
        if ma == -1000:
            ma = max(self.reward_map[r, c])
        return mi, ma

    def _show_numbers(self):
        cmap = plt.get_cmap("ValueFunction-New")
        rw_min, rw_max = self._rew_range()
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if self.reward_map[r, c] == 0:
                continue
            raw_reward = self.reward_map[r, c]
            if self.map[r, c] == self.EMPTY:
                if rw_max > rw_min:
                    reward = (self.reward_map[r, c] - rw_min) / (rw_max - rw_min)
                else:
                    reward = 0.7
                color = cmap(reward)
            elif self.map[r, c] == self.GOAL or self.PIT:
                color = "w"
            else:
                continue
            self.domain_ax.text(c - 0.2, r + 0.1, str(raw_reward), color=color)

    def _show_map(self):
        super()._show_map()
        with with_bold_fonts():
            self._show_numbers()
