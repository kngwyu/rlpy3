import click
from rlpy.domains import ChainMDP
from rlpy.tools.cli import run_experiment

import methods


def select_domain(chain_size):
    return ChainMDP(chain_size=chain_size)


def select_agent(name, domain, max_steps, seed, **kwargs):
    if name is None or name == "lspi":
        return methods.tabular_lspi(domain, max_steps)
    elif name == "nac":
        return methods.tabular_nac(domain)
    elif name == "tabular-q":
        return methods.tabular_q(domain, initial_learn_rate=0.1)
    elif name == "ifddk-q":
        return methods.ifddk_q(domain, initial_learn_rate=0.1)
    elif name == "psrl":
        return methods.tabular_psrl(domain, seed=seed)
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=10000,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[click.Option(["--chain-size"], type=int, default=4)],
    )
