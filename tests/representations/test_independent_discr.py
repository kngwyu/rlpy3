from rlpy.representations import IndependentDiscretization
from rlpy.domains import GridWorld, infinite_track_cartpole as inf_cp
import numpy as np
from rlpy.tools import __rlpy_location__
import os


def test_number_of_cells():
    """ Ensure create appropriate # of cells (despite ``discretization``) """
    mapDir = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    mapfile = os.path.join(mapDir, "4x5.txt")  # expect 4*5 = 20 states
    domain = GridWorld(mapfile=mapfile)

    rep = IndependentDiscretization(domain, discretization=100)
    assert rep.features_num == 9
    rep = IndependentDiscretization(domain, discretization=5)
    assert rep.features_num == 9


def test_phi_cells():
    """ Ensure correct features are activated for corresponding state """
    mapDir = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    mapfile = os.path.join(mapDir, "4x5.txt")  # expect 4*5 = 20 states
    domain = GridWorld(mapfile=mapfile)

    rep = IndependentDiscretization(domain)

    for r in np.arange(4):
        for c in np.arange(5):
            phiVec = rep.phi(np.array([r, c]), terminal=False)
            assert sum(phiVec) == 2  # 1 for each dimension
            assert phiVec[r] == 1  # correct row activated
            assert phiVec[4 + c] == 1  # correct col activated


def test_continuous_discr():
    """ Ensure correct discretization in continuous state spaces """
    # NOTE - if possible, test a domain with mixed discr/continuous
    domain = inf_cp.InfTrackCartPole()  # 2 continuous dims
    rep = IndependentDiscretization(domain, discretization=20)
    assert rep.features_num == 40
    rep = IndependentDiscretization(domain, discretization=50)
    assert rep.features_num == 100
