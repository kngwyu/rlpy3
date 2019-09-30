from rlpy.Domains import PuddleWorld
from rlpy.Tools import run_experiment

import methods


DOMAIN = PuddleWorld()
MAX_STEPS = 40000


def select_agent(name, seed):
    if name is None or name == "lspi":
        return methods.tabular_lspi(DOMAIN, MAX_STEPS)
    elif name == "tabular-q":
        return methods.tabular_q(DOMAIN)
    elif name == "tabular-sarsa":
        return methods.tabular_sarsa(DOMAIN)
    elif name == "ifdd-q":
        return methods.ifdd_q(
            DOMAIN,
            discretization=18,
            lambda_=0.42,
            boyan_N0=202,
            initial_learn_rate=0.7422,
        )
    elif name == "kifdd-q":
        return methods.kifdd_q(
            DOMAIN,
            8.567677,
            threshold=0.0807,
            lambda_=0.52738,
            initial_learn_rate=0.4244,
            boyan_N0=389.56,
        )
    elif name == "rbfs-q":
        return methods.rbf_q(
            DOMAIN,
            seed,
            num_rbfs=96,
            resolution=21,
            initial_learn_rate=0.6633,
            lambda_=0.1953,
            boyan_N0=13444.0,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    run_experiment(
        DOMAIN,
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=20,
        default_checks_per_policy=100,
    )
