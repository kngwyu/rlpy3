"""Nosetests for testing the domains and their methods."""
import rlpy.domains
from rlpy.domains.domain import Domain
import numpy as np
import inspect

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"


def test_random_trajectory():
    for d in list(rlpy.domains.__dict__.values()):
        if d == Domain:
            continue
        if inspect.isclass(d) and issubclass(d, Domain):
            check_random_trajectory(d)


def test_specification():
    for d in list(rlpy.domains.__dict__.values()):
        if d == Domain:
            continue
        if inspect.isclass(d) and issubclass(d, Domain):
            check_specifications(d)


def check_random_trajectory(domain_class):
    """
    run the domain 1000 steps with random actions
    Just make sure no error occurs
    """
    np.random.seed(1)
    domain = domain_class()
    terminal = True
    steps = 0
    T = 1000
    while steps < T:
        if terminal:
            s, terminal, p_actions = domain.s0()
        elif steps % domain.episode_cap == 0:
            s, terminal, p_actions = domain.s0()
        a = np.random.choice(p_actions)
        r, s, terminal, p_actions = domain.step(a)
        steps += 1


def check_specifications(domain_class):
    domain = domain_class()
    for v in ["statespace_limits", "num_actions", "episode_cap"]:
        assert getattr(domain, v) is not None
