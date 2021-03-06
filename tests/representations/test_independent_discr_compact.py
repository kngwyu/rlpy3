from rlpy.representations import IndependentDiscretizationCompactBinary
from rlpy.domains import GridWorld, SystemAdministrator
import numpy as np
from rlpy.tools import __rlpy_location__
import os


def test_number_of_cells():
    """ Ensure create appropriate # of cells (despite ``discretization``) """
    mapDir = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    mapfile = os.path.join(mapDir, "4x5.txt")  # expect 4*5 = 20 states
    domain = GridWorld(mapfile=mapfile)

    rep = IndependentDiscretizationCompactBinary(domain, discretization=100)
    assert rep.features_num == 9 + 1
    rep = IndependentDiscretizationCompactBinary(domain, discretization=5)
    assert rep.features_num == 9 + 1


def test_compact_binary():
    """ Test representation on domain with some binary dimensions """
    mapDir = os.path.join(__rlpy_location__, "domains", "SystemAdministratorMaps")
    mapname = os.path.join(mapDir, "20MachTutorial.txt")  # expect 20+1 = 21 states
    domain = SystemAdministrator(networkmapname=mapname)

    rep = IndependentDiscretizationCompactBinary(domain)
    assert rep.features_num == 21

    stateVec = np.zeros(20)
    stateVec[0] = 1

    phiVec = rep.phi(stateVec, terminal=False)

    assert sum(phiVec) == 1
    assert phiVec[0] == 1
