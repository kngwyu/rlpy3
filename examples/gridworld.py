import click
from rlpy.domains import GridWorld
from rlpy.tools.cli import run_experiment

import fr_gridworld


def select_domain(map_, noise, **kwargs):
    map_ = GridWorld.default_map(map_ + ".txt")
    return GridWorld(map_, random_start=True, noise=noise, episode_cap=20)


if __name__ == "__main__":
    run_experiment(
        select_domain,
        fr_gridworld.select_agent,
        default_max_steps=10000,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[
            click.Option(["--map", "map_"], type=str, default="4x5"),
            click.Option(["--noise"], type=float, default=0.1),
            click.Option(["--epsilon"], type=float, default=0.1),
            click.Option(["--epsilon-min"], type=float, default=None),
            click.Option(["--beta"], type=float, default=0.05),
            click.Option(["--step-penalty"], type=float, default=0.5),
            click.Option(["--episode-cap"], type=int, default=20),
            click.Option(["--vi-threshold"], type=float, default=1e-6),
            click.Option(["--show-reward"], is_flag=True),
        ],
    )
