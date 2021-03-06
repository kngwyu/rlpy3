"""Flipboard domain."""
from rlpy.tools import FONTSIZE, id2vec, plt
from .domain import Domain
import numpy as np

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


class FlipBoard(Domain):

    """
    A domain based on the last puzzle of Doors and Rooms Game stage 5-3.

    The goal of the game is to get all elements of a 4x4 board
    to have value 1.

    The initial state is the following::

        1 0 0 0
        0 0 0 0
        0 1 0 0
        0 0 1 0

    **STATE:** a 4x4 array of binary values. \n
    **ACTION:** Invert the value of a given [Row, Col] (from 0->1 or 1->0).\n
    **TRANSITION:** Determinisically flip all elements of the board on the same
    row OR col of the action. \n
    **REWARD:** -1 per step. 0 when the board is solved [all ones]
    **REFERENCE:**

    .. seealso::
        `gameday inc. Doors and Rooms game <http://bit.ly/SYqdZI>`_

    """

    BOARD_SIZE = 4
    STEP_REWARD = -1

    # Visual Stuff
    domain_fig = None
    move_fig = None

    def __init__(self):
        boards_num = self.BOARD_SIZE ** 2
        super().__init__(
            num_actions=boards_num,
            statespace_limits=np.tile([0, 1], (boards_num, 1)),
            discount_factor=1.0,
            episode_cap=min(100, boards_num),
        )

    def show_domain(self, a=0):
        s = self.state
        # Draw the environment
        if self.domain_fig is None:
            self.move_fig = plt.subplot(111)
            s = s.reshape((self.BOARD_SIZE, self.BOARD_SIZE))
            self.domain_fig = plt.imshow(
                s, cmap="FlipBoard", interpolation="nearest", vmin=0, vmax=1
            )
            plt.xticks(np.arange(self.BOARD_SIZE), fontsize=FONTSIZE)
            plt.yticks(np.arange(self.BOARD_SIZE), fontsize=FONTSIZE)
            # pl.tight_layout()
            a_row, a_col = id2vec(a, [self.BOARD_SIZE, self.BOARD_SIZE])
            self.move_fig = self.move_fig.plot(a_col, a_row, "kx", markersize=30.0)
            plt.show()
        a_row, a_col = id2vec(a, [self.BOARD_SIZE, self.BOARD_SIZE])
        self.move_fig.pop(0).remove()
        # print a_row,a_col
        # Instead of '>' you can use 'D', 'o'
        self.move_fig = plt.plot(a_col, a_row, "kx", markersize=30.0)
        s = s.reshape((self.BOARD_SIZE, self.BOARD_SIZE))
        self.domain_fig.set_data(s)
        plt.draw()

    def step(self, a):
        ns = self.state.copy()
        ns = np.reshape(ns, (self.BOARD_SIZE, -1))
        a_row, a_col = id2vec(a, [self.BOARD_SIZE, self.BOARD_SIZE])
        ns[a_row, :] = np.logical_not(ns[a_row, :])
        ns[:, a_col] = np.logical_not(ns[:, a_col])
        ns[a_row, a_col] = not ns[a_row, a_col]
        if self.is_terminal():
            terminal = True
            r = 0
        else:
            terminal = False
            r = self.STEP_REWARD
        ns = ns.flatten()
        self.state = ns.copy()
        return r, ns, terminal, self.possible_actions()

    def s0(self):
        self.state = np.array(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="bool"
        ).flatten()
        return self.state, self.is_terminal(), self.possible_actions()

    def is_terminal(self):
        return np.count_nonzero(self.state) == self.BOARD_SIZE ** 2
