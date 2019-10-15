from rlpy.domains import BicycleRiding
from rlpy.tools.cli import run_experiment

import methods


def select_agent(name, domain, max_steps, _seed):
    if name is None or name == "kifddk-q":
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
        default_max_steps=150000,
        default_num_policy_checks=30,
        default_checks_per_policy=1,
    )
