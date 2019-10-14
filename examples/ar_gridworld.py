import click
from rlpy.domains import AnyRewardGridWorld
from rlpy.tools.cli import run_experiment

import methods


def select_domain(
    map_="6x6guided", noise=0.1, step_penalty=0.5, episode_cap=20, **kwargs
):
    map_ = AnyRewardGridWorld.default_map(map_ + ".txt")
    return AnyRewardGridWorld(
        map_,
        random_start=True,
        noise=noise,
        step_penalty=step_penalty,
        episodeCap=episode_cap,
    )


def select_agent(
    name, domain, max_steps, _seed, epsilon=0.1, epsilon_min=None, **kwargs
):
    if epsilon_min is not None:
        eps_decay = (epsilon - epsilon_min) / max_steps * 0.9
        eps_min = epsilon_min
    else:
        eps_decay, eps_min = 0.0, 0.0
    if name is None or name == "lspi":
        return methods.tabular_lspi(domain, max_steps)
    elif name == "nac":
        return methods.tabular_nac(domain)
    elif name == "tabular-q":
        return methods.tabular_q(
            domain,
            epsilon=epsilon,
            epsilon_decay=eps_decay,
            epsilon_min=eps_min,
            initial_learn_rate=0.5,
        )
    elif name == "ifddk-q":
        return methods.ifddk_q(domain, epsilon=epsilon, initial_learn_rate=0.5)
    else:
        raise NotImplementedError("Method {} is not supported".format(name))


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=10000,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
        other_options=[
            click.Option(["--map", "map_"], type=str, default="6x6guided"),
            click.Option(["--noise"], type=float, default=0.1),
            click.Option(["--epsilon"], type=float, default=0.1),
            click.Option(["--epsilon-min"], type=float, default=None),
            click.Option(["--step-penalty"], type=float, default=0.5),
            click.Option(["--episode-cap"], type=int, default=20),
        ],
    )
