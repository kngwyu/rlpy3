"""BernoulliGridworld Domain."""
import itertools
import numpy as np
from rlpy.tools import __rlpy_location__, plt
import os

from .fixed_reward_grid_world import FixedRewardGridWorld

__license__ = "BSD 3-Clause"
__author__ = "Yuji Kanagawa"


class BernoulliGridWorld(FixedRewardGridWorld):
    """The same as GridWorld, but rewards are sampled from Bernoulli distributions.
    """

    # directory of maps shipped with rlpy
    DEFAULT_MAP_DIR = os.path.join(
        __rlpy_location__, "domains", "BernoulliGridWorldMaps"
    )

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
        mapfile=os.path.join(DEFAULT_MAP_DIR, "5x5normal.txt"),
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
