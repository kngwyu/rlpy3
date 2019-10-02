"""AnyRewardGridworld Domain."""
import numpy as np
from rlpy.Tools import __rlpy_location__
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
        map_and_reward = np.loadtxt(mapfile)
        mshape = map_and_reward.shape
        if mshape[1] * 2 != mshape[0]:
            raise ValueError('Invalid map with shape {}'.format(mshape))
        col = mshape[0] // 2
        self.map = map_and_reward[:col]
        self.reward_map = map_and_reward[col:]

    def __init__(
        self,
        mapfile=os.path.join(DEFAULT_MAP_DIR, "6x6guided.txt"),
        noise=0.1,
        step_penalty=0.01,
        random_start=False,
        episodeCap=1000,
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
