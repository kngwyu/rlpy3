from rlpy.representations import Tabular
from rlpy.domains import ChainMDP
from rlpy.agents import SARSA

import numpy as np

from rlpy.policies import eGreedy
from rlpy.experiments import Experiment
from .helpers import check_seed_vis


def _make_experiment(exp_id=1, path="./Results/Tmp/test_ChainMDP/"):
    ## Domain:
    chain_size = 5
    domain = ChainMDP(chain_size=chain_size)

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation = Tabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.2)

    ## Agent
    agent = SARSA(
        representation=representation,
        policy=policy,
        discount_factor=domain.discount_factor,
        initial_learn_rate=0.1,
    )
    checks_per_policy = 3
    max_steps = 50
    num_policy_checks = 3
    experiment = Experiment(**locals())
    return experiment


def test_seed():
    check_seed_vis(_make_experiment)


def test_transitions():
    """
    Ensure that actions result in expected state transition behavior.
    Note that if the agent attempts to leave the edge
    (select LEFT from s0 or RIGHT from s49) then the state should not change.
    NOTE: assume p_action_failure is only noise term.

    """
    # [[initialize domain]]
    chain_size = 5
    domain = ChainMDP(chain_size=chain_size)
    dummyS = domain.s0()
    domain.state = np.array([2])  # state s2
    left = 0
    right = 1

    # Check basic step
    r, ns, terminal, possibleA = domain.step(left)
    assert ns[0] == 1 and terminal == False
    assert np.all(possibleA == np.array([left, right]))  # all actions available
    assert r == domain.STEP_REWARD

    # Ensure all actions available, even on corner case, to meet domain specs
    r, ns, terminal, possibleA = domain.step(left)
    assert ns[0] == 0 and terminal == False
    assert np.all(possibleA == np.array([left, right]))  # all actions available
    assert r == domain.STEP_REWARD

    # Ensure state does not change or wrap around per domain spec
    r, ns, terminal, possibleA = domain.step(left)
    assert ns[0] == 0 and terminal == False
    assert np.all(possibleA == np.array([left, right]))  # all actions available
    assert r == domain.STEP_REWARD

    r, ns, terminal, possibleA = domain.step(right)
    assert ns[0] == 1 and terminal == False
    assert np.all(possibleA == np.array([left, right]))  # all actions available
    assert r == domain.STEP_REWARD

    r, ns, terminal, possibleA = domain.step(right)
    r, ns, terminal, possibleA = domain.step(right)
    r, ns, terminal, possibleA = domain.step(right)

    # Ensure goal state gives proper condition
    assert ns[0] == 4 and terminal == True
    assert np.all(possibleA == np.array([left, right]))  # all actions available
    assert r == domain.GOAL_REWARD
