import click
from rlpy.domains import BernoulliGridWorld
from rlpy.tools.cli import run_experiment

from fr_gridworld import select_agent


def select_domain(map_, noise, episode_cap, **kwargs):
    map_ = BernoulliGridWorld.default_map(map_ + ".txt")
    return BernoulliGridWorld(
        map_,
        noise=noise,
        episode_cap=episode_cap,
    )


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=10000,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[
            click.Option(["--map", "map_"], type=str, default="5x5normal"),
            click.Option(["--noise"], type=float, default=0.1),
            click.Option(["--episode-cap"], type=int, default=20),
            click.Option(["--epsilon"], type=float, default=0.1),
            click.Option(["--epsilon-min"], type=float, default=None),
            click.Option(["--beta"], type=float, default=0.05),
            click.Option(["--show-reward"], is_flag=True),
            click.Option(["--vi-threshold"], type=float, default=1e-6),
        ],
    )
