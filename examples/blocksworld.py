import click
import numpy as np
from rlpy.Domains import BlocksWorld
from rlpy.Tools.cli import run_experiment

import methods


MAX_STEPS = 100000


def select_domain(blocks=6, noise=0.3, **kwargs):
    return BlocksWorld(blocks=blocks, noise=noise)


def select_agent(name, domain, _seed, lambda_=0.1, **kwargs):
    if name is None or name == "tabular-q":
        return methods.tabular_q(domain, initial_learn_rate=0.9)
    elif name == "ifdd-ggq":
        return methods.ifdd_q(
            domain,
            lambda_=lambda_,
            boyan_N0=1220.247254,
            initial_learn_rate=0.27986823,
            ifddplus=1.0 - 1e-7,
        )
    elif name == "ifdd-q":
        return methods.ifdd_q(
            domain,
            threshold=0.03104970,
            lambda_=lambda_,
            boyan_N0=1220.247254,
            initial_learn_rate=0.27986823,
            ifddplus=1.0 - 1e-7,
        )
    elif name == "ifdd-sarsa":
        return methods.ifdd_sarsa(
            domain,
            threshold=0.023476,
            lambda_=lambda_,
            boyan_N0=20.84362,
            initial_learn_rate=0.3356222674,
            ifddplus=1.0 - 1e-7,
        )
    elif name == "tile-ggq":
        mat = np.matrix(
            """1 1 1 0 0 0;
               0 1 1 1 0 0;
               0 0 1 1 1 0;
               0 0 0 1 1 1;
               0 0 1 0 1 1;
               0 0 1 1 0 1;
               1 0 1 1 0 0;
               1 0 1 0 1 0;
               1 0 0 1 1 0;
               1 0 0 0 1 1;
               1 0 1 0 0 1;
               1 0 0 1 0 1;
               1 1 0 1 0 0;
               1 1 0 0 1 0;
               1 1 0 0 0 1;
               0 1 0 1 1 0;
               0 1 0 0 1 1;
               0 1 0 1 0 1;
               0 1 1 0 1 0;
               0 1 1 0 0 1"""
        )
        return methods.tile_ggq(
            domain,
            mat,
            lambda_=lambda_,
            initial_learn_rate=0.240155681,
            boyan_N0=14.44946,
        )
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=20,
        default_checks_per_policy=1,
        other_options=[
            click.Option(["--blocks"], type=int, default=6),
            click.Option(["--noise"], type=float, default=0.3),
            click.Option(["--lambda", "lambda_"], type=float, default=0.0),
        ],
    )
