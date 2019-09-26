import numpy as np
import pickle
from rlpy.Domains import FiniteCartPoleBalanceOriginal, InfCartPoleBalance
from rlpy.Tools import __rlpy_location__
import os
import sys
import pytest


def get_file(fname):
    return os.path.join(__rlpy_location__, '..', 'tests', fname)


@pytest.mark.parametrize('domain_class, filename', [
    (InfCartPoleBalance, get_file('traj_InfiniteCartpoleBalance.pck')),
    (FiniteCartPoleBalanceOriginal, get_file('traj_FiniteCartpoleBalanceOriginal.pck')),
])
def test_trajectory(domain_class, filename):
    with open(filename, 'rb') as f:
        if not sys.version_info[:2] == (2, 7):
            traj = pickle.load(f, encoding='latin1')
        else:
            traj = pickle.load(f)
    traj_now = sample_random_trajectory(domain_class)
    for i, e1, e2 in zip(list(range(len(traj_now))), traj_now, traj):
        print(i)
        print(e1[0], e2[0])
        if not np.allclose(e1[0], e2[0]):  # states
            print(e1[0], e2[0])
            assert False
        assert e1[-1] == e2[-1]  # reward
        print("Terminal", e1[1], e2[1])
        assert e1[1] == e2[1]  # terminal
        assert len(e1[2]) == len(e2[2])
        assert np.all([a == b for a, b in zip(e1[2], e2[2])])  # p_actions


def sample_random_trajectory(domain_class):
    """
    sample a trajectory of 1000 steps
    """
    traj = []
    np.random.seed(1)
    domain = domain_class()
    domain.random_state = np.random.RandomState(1)
    terminal = True
    steps = 0
    T = 1000
    r = 0
    while steps < T:
        if terminal:
            s, terminal, p_actions = domain.s0()
        elif steps % domain.episodeCap == 0:
            s, terminal, p_actions = domain.s0()
        a = np.random.choice(p_actions)
        traj.append((s, terminal, p_actions, a, r))
        r, s, terminal, p_actions = domain.step(a)
        steps += 1
    return traj
