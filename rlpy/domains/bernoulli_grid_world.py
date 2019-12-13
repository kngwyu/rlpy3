"""BernoulliGridworld Domain."""
import itertools
import numpy as np
from rlpy.tools import __rlpy_location__, plt
from pathlib import Path

from .fixed_reward_grid_world import FixedRewardGridWorld

__license__ = "BSD 3-Clause"
__author__ = "Yuji Kanagawa"


class BernoulliGridWorld(FixedRewardGridWorld):
    """The same as GridWorld, but rewards are sampled from Bernoulli distributions.
    """

    # directory of maps shipped with rlpy
    DEFAULT_MAP_DIR = Path(__rlpy_location__).joinpath("domains/BernoulliGridWorldMaps")

    def _load_map(self, mapfile):
        map_and_prob = np.loadtxt(mapfile, dtype=np.float64)
        mshape = map_and_prob.shape
        if mshape[1] * 2 != mshape[0]:
            raise ValueError("Invalid map with shape {}".format(mshape))
        col = mshape[0] // 2
        self.prob_map = map_and_prob[col:]
        if (self.prob_map < 0).any() or (1 < self.prob_map).any():
            raise ValueError(
                "Map for BernoulliRewardGridWorld contains invalid probability value!"
            )
        return map_and_prob[:col].astype(np.int32)

    def __init__(
        self,
        mapfile=DEFAULT_MAP_DIR.joinpath("5x5normal.txt"),
        noise=0.1,
        random_start=False,
        episode_cap=20,
    ):
        super().__init__(
            mapfile=mapfile,
            noise=noise,
            random_start=random_start,
            episode_cap=episode_cap,
        )

    def _reward(self, next_state, terminal):
        prob = self.prob_map[next_state[0], next_state[1]]
        reward = self.random_state.binomial(1, prob)
        if not terminal:
            reward += self.STEP_REWARD
        return reward

    def _show_numbers(self):
        cmap = plt.get_cmap("ValueFunction-New")
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if self.prob_map[r, c] == 0:
                continue
            prob = self.prob_map[r, c]
            if self.map[r, c] == self.EMPTY:
                color = cmap(prob)
            elif self.map[r, c] == self.GOAL or self.PIT:
                color = "w"
            else:
                continue
            self.domain_ax.text(c - 0.2, r + 0.1, str(prob), color=color)

    def expected_step(self, s, a):
        raise NotImplementedError()
