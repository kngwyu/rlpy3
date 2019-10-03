from rlpy.Domains import BicycleRiding
from rlpy.Tools import run_experiment

import methods


MAX_STEPS = 150000


def select_agent(name, domain, _seed):
    if name is None or name == "lspi":
        return methods.tabular_lspi(domain, MAX_STEPS)
    elif name == "nac":
        return methods.tabular_nac(domain)
    elif name == "tabular-q":
        return methods.tabular_q(domain, initial_learn_rate=0.9)
    elif name == "kifddk-q":
        return methods.kifdd_q(
            domain,
            11.6543336229,
            threshold=88044,
            lambda_=0.43982644088,
            initial_learn_rate=0.920244401,
            boyan_N0=64502.0,
            kernel="linf_triangle_kernel",
        )
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        BicycleRiding(),
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=30,
        default_checks_per_policy=1,
    )
