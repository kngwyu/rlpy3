from rlpy.Domains import PuddleWorld
from rlpy.Tools.cli import run_experiment

import methods


def select_agent(name, domain, max_steps, seed):
    if name is None or name == "lspi":
        return methods.tabular_lspi(domain, max_steps)
    elif name == "tabular-q":
        return methods.tabular_q(domain)
    elif name == "tabular-sarsa":
        return methods.tabular_sarsa(domain)
    elif name == "ifdd-q":
        return methods.ifdd_q(
            domain,
            discretization=18,
            lambda_=0.42,
            boyan_N0=202,
            initial_learn_rate=0.7422,
        )
    elif name == "kifdd-q":
        return methods.kifdd_q(
            domain,
            8.567677,
            threshold=0.0807,
            lambda_=0.52738,
            initial_learn_rate=0.4244,
            boyan_N0=389.56,
        )
    elif name == "rbfs-q":
        return methods.rbf_q(
            domain,
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
        PuddleWorld(),
        select_agent,
        default_max_steps=40000,
        default_num_policy_checks=20,
        default_checks_per_policy=100,
    )
