"""
This is only for testing.
Lifegame is too difiicult for tabular methods.
"""
import click
from rlpy.domains import LifeGameSurvival
from rlpy.tools.cli import run_experiment

from fr_gridworld import select_agent


def select_domain(rule, init, episode_cap, **kwargs):
    path = LifeGameSurvival.DEFAULT_MAP_DIR
    if not init.endswith(".txt"):
        init = init + ".txt"
    init_file = path.joinpath(rule).joinpath(init)
    return LifeGameSurvival(init_file, rule=rule, episode_cap=episode_cap)


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=10000,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[
            click.Option(["--rule"], type=str, default="life"),
            click.Option(["--init"], type=str, default="7x7ever"),
            click.Option(["--episode-cap"], type=int, default=100),
            click.Option(["--epsilon"], type=float, default=0.1),
            click.Option(["--epsilon-min"], type=float, default=None),
            click.Option(["--beta"], type=float, default=0.05),
            click.Option(["--show-reward"], is_flag=True),
            click.Option(["--vi-threshold"], type=float, default=1e-6),
        ],
    )
