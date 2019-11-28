import click
from rlpy.domains import (
    FiniteCartPoleBalance,
    FiniteCartPoleBalanceOriginal,
    FiniteCartPoleBalanceModern,
    FiniteCartPoleSwingUp,
    FiniteCartPoleSwingUpFriction,
)
from rlpy.tools.cli import run_experiment

import methods

DOMAINS = {
    "balance": FiniteCartPoleBalance,
    "original": FiniteCartPoleBalanceOriginal,
    "modern": FiniteCartPoleBalanceModern,
    "swingup": FiniteCartPoleSwingUp,
    "friction": FiniteCartPoleSwingUpFriction,
}


def select_domain(task, good_reward, **kwargs):
    if task == "original":
        return DOMAINS[task](good_reward)
    else:
        return DOMAINS[task]()


def select_agent(name, domain, max_steps, seed, **kwargs):
    if name is None or name == "ifdd-q":
        return methods.ifdd_q(
            domain,
            discretization=47,
            threshold=77.0,
            lambda_=0.9,
            initial_learn_rate=0.05,
            boyan_N0=11,
            ifddplus=True,
        )
    elif name == "kifdd-q":
        return methods.kifdd_q(
            domain,
            kernel_resolution=13.14,
            threshold=0.21,
            lambda_=0.9,
            initial_learn_rate=0.07,
            boyan_N0=37.0,
            kernel="gaussian_kernel",
        )
    elif name == "tabular-q":
        return methods.tabular_q(
            domain,
            lambda_=0.9,
            initial_learn_rate=0.26,
            boyan_N0=119,
            incremental=True,
        )
    elif name == "rbf-q":
        return methods.rbf_q(
            domain,
            seed,
            num_rbfs=5000,
            resolution=8,
            initial_learn_rate=0.26,
            lambda_=0.9,
            boyan_N0=2120,
        )
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=30000,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[
            click.Option(["--task"], type=str, default="original"),
            click.Option(["--good-reward"], type=float, default=0.0),
            click.Option(["--kernel"], type=str, default="gaussian_kernel"),
        ],
    )
