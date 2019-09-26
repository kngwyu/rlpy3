from rlpy.Representations import Tabular
from rlpy.Domains.InfiniteTrackCartPole import InfCartPoleBalance, InfCartPoleSwingUp
from rlpy.Agents.TDControlAgent import SARSA
import numpy as np
from .helpers import check_seed_vis
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment


def _make_experiment(domain, exp_id=1, \
                     path="./Results/Tmp/test_InfTrackCartPole"):
    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.2)

    ## Agent
    agent = SARSA(representation=representation, policy=policy,
                  discount_factor=domain.discount_factor,
                       learn_rate=0.1)
    checks_per_policy = 3
    max_steps = 50
    num_policy_checks = 3
    experiment = Experiment(**locals())
    return experiment


def test_seed_balance():
    """ Ensure that providing the same random seed yields same result """

    def myfn(*args, **kwargs):
        return _make_experiment(InfCartPoleBalance(), *args, **kwargs)
    check_seed_vis(myfn)


def test_seed_swingup():
    def myfn(*args, **kwargs):
        return _make_experiment(InfCartPoleSwingUp(), *args, **kwargs)
    check_seed_vis(myfn)


def test_physicality():
    """
    Test coordinate system [vertical is 0]
        1) gravity acts in proper direction based on origin
        2) torque actions behave as expected in that frame
    """
    # Apply a bunch of non-torque actions, ensure that monotonically accelerate

    LEFT_FORCE = 0
    NO_FORCE = 1
    RIGHT_FORCE = 2

    domain = InfCartPoleSwingUp()
    domain.force_noise_max = 0 # no stochasticity in applied FORCE

    # Positive angle (right)
    s = np.array([10.0 * np.pi/180.0, 0.0]) # pendulum slightly right
    domain.state = s.copy()

    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.all(domain.state > s) # angle and angular velocity increase
        s = domain.state.copy()

    # Negative angle (left)
    s = np.array([-10.0 * np.pi/180.0, 0.0]) # pendulum slightly right
    domain.state = s.copy()

    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.all(domain.state < s) # angle and angular velocity increase
        s = domain.state.copy()

    # Ensure that reward racks up while in region

def test_physicality_hanging():
    """
    Test that energy does not spontaneously enter system
    """
    # Apply a bunch of non-torque actions, ensure that monotonically accelerate

    LEFT_FORCE = 0
    NO_FORCE = 1
    RIGHT_FORCE = 2

    domain = InfCartPoleSwingUp()
    domain.force_noise_max = 0 # no stochasticity in applied FORCE

    # Positive angle (right)
    s = np.array([179.6 * np.pi/180.0, 0.0]) # pendulum hanging down
    domain.state = s

    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.abs(domain.state[0]) <=179.5 # angle does not increase
        assert np.abs(domain.state[1]) <= 0.1 # angular rate does not increase
        s = domain.state
