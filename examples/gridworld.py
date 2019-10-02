from rlpy.Domains import GridWorld
from rlpy.Tools import run_experiment

import methods


DOMAIN = GridWorld(GridWorld.default_map("4x5.txt"), random_start=True, noise=0.3)
MAX_STEPS = 10000


def select_agent(name, _seed):
    if name is None or name == "lspi":
        return methods.tabular_lspi(DOMAIN, MAX_STEPS)
    elif name == "nac":
        return methods.tabular_nac(DOMAIN)
    elif name == "tabular-q":
        return methods.tabular_q(DOMAIN, initial_learn_rate=0.11)
    elif name == "ifddk-q":
        return methods.tabular_q(DOMAIN, initial_learn_rate=0.11)
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        DOMAIN,
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
    )
