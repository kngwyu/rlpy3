from rlpy.representations import OMPTD, IndependentDiscretization
from rlpy.domains import GridWorld
import numpy as np
from rlpy.tools import __rlpy_location__
import os


def test_bag_creation():
    """
    Ensure create appropriate # of conjunctions, that they have been
    instantiated properly, and there are no duplicates.
    """
    mapDir = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    mapfile = os.path.join(mapDir, "4x5.txt")  # expect 4*5 = 20 states
    domain = GridWorld(mapfile=mapfile)

    initial_representation = IndependentDiscretization(domain)
    max_batch_discovery = np.inf
    batch_threshold = 1e-10
    discretization = 20
    bag_size = 100000  # We add all possible features

    rep = OMPTD(
        domain,
        initial_representation,
        discretization,
        max_batch_discovery,
        batch_threshold,
        bag_size,
        sparsify=False,
    )
    assert rep.total_feature_size == 9 + 20
    assert rep.features_num == 9

    # Compute full (including non-discovered) feature vec for a few states
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    rep.calculate_full_phi_normalized(states)
    phi_states = rep.fullphi
    phi_states[phi_states > 0] = True
    true_phi_s1 = np.zeros(len(phi_states[0, :]))
    true_phi_s1[0] = True
    true_phi_s1[4] = True  # TODO - could be [4] depending on axes, check.
    true_phi_s1[9] = True  # The conjunction of [0,0]
    assert np.all(true_phi_s1 == phi_states[0, :])  # expected feature vec returned
    assert sum(phi_states[0, :]) == 3  # 2 original basic feats and 1 conjunction

    true_phi_s2 = np.zeros(len(phi_states[0, :]))
    true_phi_s2[0] = True
    true_phi_s2[5] = True  # TODO - could be [4] depending on axes, check.
    true_phi_s2[10] = True  # The conjunction of [0,0]
    assert np.all(true_phi_s2 == phi_states[1, :])  # expected feature vec returned
    assert sum(phi_states[1, :]) == 3  # 2 original basic feats and 1 conjunction

    true_phi_s3 = np.zeros(len(phi_states[0, :]))
    true_phi_s3[1] = True
    true_phi_s3[4] = True  # TODO - could be [4] depending on axes, check.
    true_phi_s3[14] = True  # The conjunction of [0,0]
    assert np.all(true_phi_s3 == phi_states[2, :])  # expected feature vec returned
    assert sum(phi_states[2, :]) == 3  # 2 original basic feats and 1 conjunction

    true_phi_s4 = np.zeros(len(phi_states[0, :]))
    true_phi_s4[1] = True
    true_phi_s4[5] = True  # TODO - could be [4] depending on axes, check.
    true_phi_s4[15] = True  # The conjunction of [0,0]
    assert np.all(true_phi_s4 == phi_states[3, :])  # expected feature vec returned
    assert sum(phi_states[3, :]) == 3  # 2 original basic feats and 1 conjunction


def test_batch_discovery():
    """
    Test feature discovery from features available in bag, and that appropriate
    feats are activiated in later calls to phi_nonterminal()
    """
    mapDir = os.path.join(__rlpy_location__, "domains", "GridWorldMaps")
    mapfile = os.path.join(mapDir, "4x5.txt")  # expect 4*5 = 20 states
    domain = GridWorld(mapfile=mapfile)

    initial_representation = IndependentDiscretization(domain)
    max_batch_discovery = np.inf
    batch_threshold = 1e-10
    discretization = 20
    bag_size = 100000  # We add all possible features

    rep = OMPTD(
        domain,
        initial_representation,
        discretization,
        max_batch_discovery,
        batch_threshold,
        bag_size,
        sparsify=False,
    )
    states = np.array([[0, 0], [0, 2]])
    activePhi_s1 = rep.phi_non_terminal(states[0, :])
    activePhi_s2 = rep.phi_non_terminal(states[1, :])
    phiMatr = np.zeros((2, len(activePhi_s1)))
    phiMatr[0, :] = activePhi_s1
    phiMatr[1, :] = activePhi_s2
    td_errors = np.array([2, 5])
    flagAddedFeat = rep.batch_discover(td_errors, phiMatr, states)
    assert flagAddedFeat  # should have added at least one
    assert rep.selected_features[-1] == 9  # feat conj that yields state [0,2]
    assert rep.selected_features[-2] == 11  # feat conj that yields state [0,0]

    # Ensure that discovered features are now active
    true_phi_s1 = np.zeros(rep.features_num)
    true_phi_s1[0] = True
    true_phi_s1[4] = True  # TODO - could be [4] depending on axes, check.
    true_phi_s1[10] = True  # The conjunction of [0,0]
    assert np.all(true_phi_s1 == rep.phi_non_terminal(states[0, :]))

    true_phi_s2 = np.zeros(rep.features_num)
    true_phi_s2[0] = True
    true_phi_s2[6] = True  # TODO - could be [4] depending on axes, check.
    # The conjunction of [0,2] [[note actual id is 11, but in index 10]]
    true_phi_s2[9] = True
    assert np.all(true_phi_s2 == rep.phi_non_terminal(states[1, :]))
