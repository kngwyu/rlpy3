import numpy as np
from rlpy.representations import IncrementalTabular
from rlpy.domains.finite_track_cartpole import (
    FiniteCartPoleBalance,
    FiniteCartPoleSwingUp,
    FiniteCartPoleBalanceModern,
)
from rlpy.agents import SARSA
from rlpy.policies import eGreedy
from rlpy.experiments import Experiment
from .helpers import check_seed_vis


def _make_experiment(domain, exp_id=1, path="./Results/Tmp/test_FiniteTrackCartPole"):
    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation = IncrementalTabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.2)

    ## Agent
    agent = SARSA(
        representation=representation,
        policy=policy,
        discount_factor=domain.discount_factor,
        initial_learn_rate=0.1,
    )
    checks_per_policy = 2
    max_steps = 30
    num_policy_checks = 2
    experiment = Experiment(**locals())
    return experiment


def test_seed_balance():
    """ Ensure that providing the same random seed yields same result """

    def myfn(*args, **kwargs):
        return _make_experiment(FiniteCartPoleBalance(), *args, **kwargs)

    check_seed_vis(myfn)


def test_seed_swingup():
    def myfn(*args, **kwargs):
        return _make_experiment(FiniteCartPoleSwingUp(), *args, **kwargs)

    check_seed_vis(myfn)


def test_physicality():
    """
    Test coordinate system [vertical up is 0]
        1) gravity acts in proper direction based on origin
        2) force actions behave as expected in that frame
    """
    # Apply a bunch of non-force actions, ensure that monotonically accelerate

    LEFT_FORCE = 0
    NO_FORCE = 1
    RIGHT_FORCE = 2
    domain = FiniteCartPoleBalanceModern()
    domain.force_noise_max = 0  # no stochasticity in applied force

    domain.int_type = "rk4"

    # Slightly positive angle, just right of vertical up
    s = np.array([10.0 * np.pi / 180.0, 0.0, 0.0, 0.0])  # pendulum slightly right
    domain.state = s

    for i in np.arange(5):  # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.all(domain.state[0:2] > s[0:2])  # angle and angular velocity increase
        # no energy should enter or leave system under no force action
        assert (
            np.abs(_cartPoleEnergy(domain, s) - _cartPoleEnergy(domain, domain.state))
            < 0.01
        )

        s = domain.state

    # Negative angle (left)
    s = np.array([-10.0 * np.pi / 180.0, 0.0, 0.0, 0.0])  # pendulum slightly right
    domain.state = s

    for i in np.arange(5):  # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.all(domain.state[0:2] < s[0:2])  # angle and angular velocity increase
        # no energy should enter or leave system under no force action
        assert (
            np.abs(_cartPoleEnergy(domain, s) - _cartPoleEnergy(domain, domain.state))
            < 0.01
        )
        s = domain.state

    # Start vertical, ensure that force increases angular velocity in direction
    # Negative force on cart, yielding positive rotation
    s = np.array([0.0, 0.0, 0.0, 0.0])
    domain.state = s

    for i in np.arange(5):  # do for 5 steps and ensure works
        domain.step(LEFT_FORCE)
        assert np.all(domain.state[0:2] > s[0:2])  # angle and angular velocity increase
        s = domain.state

    # Positive force on cart, yielding negative rotation
    s = np.array([0.0, 0.0, 0.0, 0.0])
    domain.state = s

    for i in np.arange(5):  # do for 5 steps and ensure works
        domain.step(RIGHT_FORCE)
        assert np.all(domain.state[0:2] < s[0:2])  # angle and angular velocity increase
        s = domain.state
    # Ensure that reward racks up while in region


def test_physicality_hanging():
    """
    Test that energy does not spontaneously enter system
    """
    # Apply a bunch of non-force actions, ensure that monotonically accelerate

    LEFT_FORCE = 0
    NO_FORCE = 1
    RIGHT_FORCE = 2
    domain = FiniteCartPoleBalanceModern()
    domain.force_noise_max = 0  # no stochasticity in applied force
    domain.ANGLE_LIMITS = [-np.pi, np.pi]  # We actually want to test hanging
    # Positive angle (right)
    s = np.array([179.6 * np.pi / 180.0, 0.0, -2.0, 0.0])  # pendulum hanging down
    domain.state = s.copy()

    for i in np.arange(5):  # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.abs(domain.state[1]) <= 0.1  # angular rate does not increase
        # no energy should enter or leave system under no force action
        assert (
            np.abs(_cartPoleEnergy(domain, s) - _cartPoleEnergy(domain, domain.state))
            < 0.01
        )
        s = domain.state

    # Ensure that running out of x bounds causes experiment to terminate
    assert domain.is_terminal(s=np.array([0.0, 0.0, 2.5, 0.0]))
    assert domain.is_terminal(s=np.array([0.0, 0.0, -2.5, 0.0]))


def _cartPoleEnergy(domain, s):
    """
    energy equation:
    http://robotics.itee.uq.edu.au/~metr4202/tpl/t10-Week12-pendulum.pdf

    """
    cartEnergy = 0.5 * domain.MASS_CART * s[3] ** 2
    pendEnergy = (
        0.5
        * domain.MASS_PEND
        * (
            (s[2] + domain.LENGTH * np.sin(s[1])) ** 2
            + (domain.LENGTH * np.cos(s[1])) ** 2
        )
    )
    pendEnergy = pendEnergy + domain.MASS_PEND * 9.81 * domain.LENGTH * np.cos(s[0])

    return cartEnergy + pendEnergy
