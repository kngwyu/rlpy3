import click
from rlpy.domains import DeepSea
from rlpy.mdp_solvers import (
    TrajectoryBasedPolicyIteration,
    TrajectoryBasedValueIteration,
    PolicyIteration,
    ValueIteration,
)
from rlpy.representations import Tabular
from rlpy.tools.cli import run_mb_experiment


def select_domain(size, noise, **kwargs):
    return DeepSea(size, noise=noise)


def select_agent(name, domain, seed, threshold, **kwargs):
    name = None if name is None else name.lower()
    tabular = Tabular(domain, discretization=20)
    ag_kwargs = {"convergence_threshold": threshold}
    if name is None or name == "vi":
        return ValueIteration(seed, tabular, domain, **ag_kwargs)
    elif name == "pi":
        return PolicyIteration(seed, tabular, domain, **ag_kwargs)
    elif name in ["tpi", "traj-pi"]:
        return TrajectoryBasedPolicyIteration(seed, tabular, domain, **ag_kwargs)
    elif name in ["tvi", "traj-vi"]:
        return TrajectoryBasedValueIteration(seed, tabular, domain, **ag_kwargs)
    else:
        raise ValueError("{} is not supported".format(name))


if __name__ == "__main__":
    run_mb_experiment(
        select_domain,
        select_agent,
        other_options=[
            click.Option(["--size"], type=int, default=10),
            click.Option(["--noise"], type=float, default=0.0),
            click.Option(["--threshold"], type=float, default=1e-12),
        ],
    )
