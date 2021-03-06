from rlpy.representations import IncrementalTabular
from rlpy.domains import GridWorld
import numpy as np
from rlpy.tools import __rlpy_location__
import os


def test_cell_expansion():
    """ Ensure start with 0 cells, add one for each state uniquely. """
    mapDir = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    mapfile = os.path.join(mapDir, "4x5.txt")  # expect 4*5 = 20 states
    domain = GridWorld(mapfile=mapfile)

    rep = IncrementalTabular(domain, discretization=100)
    assert rep.features_num == 0  # start with 0 cells
    sOrigin = np.array([0, 0])
    s2 = np.array([1, 2])
    terminal = False  # nonterminal state
    a = 1  # arbitrary action

    # Expect to add feats for these newly seen states
    numAdded = rep.pre_discover(sOrigin, terminal, a, s2, terminal)
    assert numAdded == 2
    assert rep.features_num == 2
    phiVecOrigin = rep.phi(sOrigin, terminal)
    phiVec2 = rep.phi(s2, terminal)
    assert sum(phiVecOrigin) == 1
    assert sum(phiVec2) == 1
    phiVecOrigin2 = rep.phi(np.array([0, 0]), terminal=False)
    assert rep.features_num == 2  # didn't duplicate the feature
    assert sum(phiVecOrigin2) == 1

    # Make sure we dont duplicate feats anywhere
    numAdded = rep.pre_discover(np.array([0, 0]), terminal, a, s2, terminal)
    assert numAdded == 0
    assert rep.features_num == 2
