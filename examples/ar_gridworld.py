import click
from rlpy.Domains import AnyRewardGridWorld
from rlpy.Tools.cli import run_experiment

import methods

MAX_STEPS = 10000


def select_domain(
    map_="6x6guided", noise=0.1, step_penalty=1.0, episode_cap=20, **kwargs
):
    map_ = AnyRewardGridWorld.default_map(map_ + ".txt")
    return AnyRewardGridWorld(
        map_,
        random_start=True,
        noise=noise,
        step_penalty=step_penalty,
        episodeCap=episode_cap,
    )


def select_agent(name, domain, _seed, epsilon=0.1, **kwargs):
    if name is None or name == "lspi":
        return methods.tabular_lspi(domain, MAX_STEPS)
    elif name == "nac":
        return methods.tabular_nac(domain)
    elif name == "tabular-q":
        return methods.tabular_q(domain, epsilon=epsilon, initial_learn_rate=0.5)
    elif name == "ifddk-q":
        return methods.tabular_q(domain, epsilon=epsilon, initial_learn_rate=0.5)
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[
            click.Option(["--map", "map_"], type=str, default="6x6guided"),
            click.Option(["--noise"], type=float, default=0.1),
            click.Option(["--epsilon"], type=float, default=0.1),
            click.Option(["--step-penalty"], type=float, default=1.0),
            click.Option(["--episode-cap"], type=int, default=20),
        ],
    )
