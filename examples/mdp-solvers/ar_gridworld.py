import click
from rlpy.domains import AnyRewardGridWorld
from rlpy.mdp_solvers import (
    TrajectoryBasedPolicyIteration,
    TrajectoryBasedValueIteration,
    PolicyIteration,
    ValueIteration,
)
from rlpy.representations import Tabular
from rlpy.tools.cli import run_mb_experiment


def select_domain(map_, step_penalty, **kwargs):
    map_ = AnyRewardGridWorld.default_map(map_ + ".txt")
    return AnyRewardGridWorld(
        map_, random_start=True, noise=0.1, step_penalty=step_penalty
    )


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
            click.Option(["--map", "map_"], type=str, default="6x6guided"),
            click.Option(["--noise"], type=float, default=0.1),
            click.Option(["--threshold"], type=float, default=1e-12),
            click.Option(["--step-penalty"], type=float, default=0.5),
        ],
    )
