import click
from rlpy.domains import GridWorld
from rlpy.mdp_solvers import (
    TrajectoryBasedPolicyIteration,
    TrajectoryBasedValueIteration,
    PolicyIteration,
    ValueIteration,
)
from rlpy.representations import Tabular
from rlpy.tools.cli import run_solver_experiment


def select_domain(map_="4x5", **kwargs):
    map_ = GridWorld.default_map(map_ + ".txt")
    return GridWorld(map_, random_start=True, noise=0.1)


def select_agent(name, domain, seed, **kwargs):
    name = None if name is None else name.lower()
    tabular = Tabular(domain, discretization=20)
    if name is None or name == "vi":
        return ValueIteration(seed, tabular, domain)
    elif name == "pi":
        return PolicyIteration(seed, tabular, domain)
    elif name in ["tpi", "traj-pi"]:
        return TrajectoryBasedPolicyIteration(seed, tabular, domain)
    elif name in ["tvi", "traj-vi"]:
        return TrajectoryBasedValueIteration(seed, tabular, domain)
    else:
        raise ValueError("{} is not supported".format(name))


if __name__ == "__main__":
    run_solver_experiment(
        select_domain,
        select_agent,
        other_options=[
            click.Option(["--map", "map_"], type=str, default="4x5"),
            click.Option(["--noise"], type=float, default=0.1),
        ],
    )
