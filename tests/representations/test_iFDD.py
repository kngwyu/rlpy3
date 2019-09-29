from rlpy.Representations import iFDD
from rlpy.Representations import IndependentDiscretizationCompactBinary
from rlpy import Domains
import numpy as np

STDOUT_FILE = "out.txt"
JOB_ID = 1
RANDOM_TEST = 0
sparsify = True
discovery_threshold = 1


def test_deterministic():
    discovery_threshold = 1
    sparsify = True
    domain = Domains.SystemAdministrator()
    initialRep = IndependentDiscretizationCompactBinary(domain)
    rep = iFDD(
        domain, discovery_threshold, initialRep, debug=0, useCache=1, sparsify=sparsify
    )
    rep.theta = np.arange(rep.features_num * domain.actions_num) * 10
    print("Initial [0,1] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 1]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([0, 1]))
    # rep.show()

    print(rep.inspectPair(0, 1, discovery_threshold + 1))
    # rep.show()
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 1]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([21]))

    print("Initial [2,3] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([2, 3]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([2, 3]))
    # rep.showCache()
    # rep.showFeatures()
    # rep.showCache()
    print("Initial [0,20] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 20]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([0, 20]))

    print("Initial [0,1,20] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 1, 20]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([20, 21]))
    rep.showCache()
    # Change the weight for new feature 40
    rep.theta[40] = -100
    print("Initial [0,20] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 20]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([0, 20]))

    print("discover 0,1,20")
    rep.inspectPair(20, rep.features_num - 1, discovery_threshold + 1)
    # rep.showFeatures()
    # rep.showCache()
    print("Initial [0,1,20] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 1, 20]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([22]))

    rep.showCache()
    print("Initial [0,1,2,3,4,5,6,7,8,20] => ", end=" ")
    ANSWER = np.sort(rep.findFinalActiveFeatures([0, 1, 2, 3, 4, 5, 6, 7, 8, 20]))
    print(ANSWER)
    assert np.array_equal(ANSWER, np.array([2, 3, 4, 5, 6, 7, 8, 22]))
