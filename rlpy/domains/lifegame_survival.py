from abc import ABC, abstractmethod
import itertools
import numpy as np
from rlpy.tools import __rlpy_location__
from pathlib import Path

from .grid_world import GridWorld


__license__ = "BSD 3-Clause"
__author__ = "Yuji Kanagawa"


class LifeLike(ABC):
    """Rule of life-like cellular automaton
    """

    @abstractmethod
    def __call__(self, current, neighbors):
        pass


class LifeGame(LifeLike):
    def __call__(self, current, active_neighbors):
        if current:
            return 2 <= active_neighbors <= 3
        else:
            return active_neighbors == 3


class DryLife(LifeLike):
    def __call__(self, current, active_neighbors):
        if current:
            return 2 <= active_neighbors <= 3
        else:
            return active_neighbors in [3, 7]


class Seeds(LifeLike):
    def __call__(self, current, active_neighbors):
        if current:
            return False
        else:
            return active_neighbors == 2


RULE_REGISTORY = {
    "life": LifeGame,
    "dry": DryLife,
    "seeds": Seeds,
}


class LifeGameSurvival(GridWorld):
    """My original game. Agent survives in the famous conway's life game.
    **NOTE**
    This is not MDP with ``Domain`` API.
    Need to use gym API.
    """

    # directory of maps shipped with rlpy
    DEFAULT_MAP_DIR = Path(__rlpy_location__).joinpath("domains/LifeGameInit")
    NEIGHBORS = [-1, 0, 1]
    MAP_CATEGORY = 3

    def _load_map(self, mapfile):
        map_ = np.loadtxt(mapfile, dtype=np.uint8)
        if map_.ndim == 1:
            map_ = np.expand_dims(map_, 0)
        return map_

    def __init__(
        self,
        mapfile=DEFAULT_MAP_DIR.joinpath("life/7x7ever.txt"),
        rule="life",
        episode_cap=100,
        collison_penalty=1.0,
        survive_reward=0.01,
    ):
        super().__init__(
            mapfile=mapfile,
            noise=0.0,
            random_start=True,
            random_goal=False,
            episode_cap=episode_cap,
        )
        self.collison_penalty = collison_penalty
        self.survive_reward = survive_reward
        self._init = self.map.copy()
        self._active_cells = self.map == self.PIT
        try:
            self._rule = RULE_REGISTORY[rule]()
        except KeyError:
            raise NotImplementedError(f"Rule {rule} is not implemented")

    def s0(self):
        self.state = self._sample_start()
        self.map = self._init.copy()
        self._active_cells = self.map == self.PIT
        return self._get_obs(self.state), self.is_terminal(), self.possible_actions()

    def _cycle(self, state):
        y, x = state
        return (y + self.rows) % self.rows, (x + self.cols) % self.cols

    def _step_lifegame(self):
        next_ = np.zeros_like(self._active_cells)
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            active_neighbors = 0
            for rplus, cplus in itertools.product(self.NEIGHBORS, self.NEIGHBORS):
                if rplus == 0 and cplus == 0:
                    continue
                neighbor = self._cycle((r + rplus, c + cplus))
                if self._active_cells[neighbor]:
                    active_neighbors += 1
            next_[r, c] = self._rule(self._active_cells[r, c], active_neighbors)
        self._active_cells = next_
        for r, c in itertools.product(range(self.rows), range(self.cols)):
            if next_[r, c]:
                self.map[r, c] = self.PIT
            else:
                self.map[r, c] = self.EMPTY

    def is_terminal(self, s=None):
        if s is None:
            s = self.state
        r, c = s
        return self.map[r, c] == self.PIT

    def step(self, a):
        ns = self.state.copy()
        if self.random_state.random_sample() < self.noise:
            # Random Move
            a = self.random_state.choice(self.possible_actions())

        # Take action
        self.state = np.array(self._cycle(self.state + self.ACTIONS[a]))
        self._step_lifegame()
        self._map_changed = True

        # Check collison
        terminal = self.is_terminal()
        if terminal:
            reward = -self.collison_penalty
        else:
            reward = self.survive_reward
        return reward, self._get_obs(ns), terminal, self.possible_actions()

    def get_binary_image(self, state):
        empty = np.bitwise_or(self.map == self.EMPTY, self.map == self.START)
        pit = self.map == self.PIT
        agent = np.zeros_like(pit)
        agent[state[0], state[1]] = 1
        return np.stack([empty, pit, agent])
