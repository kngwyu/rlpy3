"""AnyRewardGridworld Domain."""
import itertools
import numpy as np
from rlpy.Tools import __rlpy_location__, mpl, plt
import os

from .GridWorld import GridWorld

__license__ = "BSD 3-Clause"
__author__ = "Yuji Kanagawa"


def _star():
    to_rad = np.pi / 180

    def rotm(angle):
        th = angle * to_rad
        return np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

    def pentagon(unit):
        return np.array([rotm(i * 72).dot(unit) for i in range(5)])

    p1 = pentagon(np.array([0.0, 0.5]))
    p2 = pentagon(np.array([0.0, 0.5 * 0.5 * np.cos(36 * to_rad)])) * -1
    res = []
    for i in range(5):
        res += [p1[i], p2[(i + 3) % 5]]
    return np.array(res)


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

    def s0(self):
        self.state = self._sample_start()
        return self.state, self.isTerminal(), self.possibleActions()

    def _reward(self, next_state, terminal):
        reward = self.reward_map[next_state[0], next_state[1]]
        if not terminal:
            reward -= self.step_penalty
        return reward

    def _show_map(self):
        self.domain_ax.imshow(
            self.map, cmap="GridWorld", interpolation="nearest", vmin=0, vmax=5
        )
        cmap = plt.get_cmap("ValueFunction-New")
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if self.map[r, c] != self.EMPTY or self.reward_map[r, c] == 0:
                continue
            reward = (self.reward_map[r, c] + 10) / 20
            star = -_star() * 0.8
            for i in range(star.shape[0]):
                star[i][0] += c
                star[i][1] += r
            patch = mpl.patches.Polygon(star, color=cmap(reward))
            self.domain_ax.add_patch(patch)
        self.domain_ax.plot([0.0], [0.0], color=cmap(1.0), label="+ reward")
        self.domain_ax.plot([0.0], [0.0], color=cmap(0.0), label="- reward")
        self.domain_ax.legend(fontsize=12)
