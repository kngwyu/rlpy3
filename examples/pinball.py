from rlpy.domains import Pinball
from rlpy.tools.cli import run_experiment

import methods


def select_domain(noise=0.1):
    return Pinball(noise=noise)


def select_agent(name, domain, max_steps, seed):
    if name is None or name == "fourier-q":
        return methods.fourier_q(domain, order=5)
    elif name == "fourier-sarsa":
        return methods.fourier_sarsa(domain, order=5)
    elif name == "ifdd-q":
        return methods.ifdd_q(domain)
    elif name == "ifdd-sarsa":
        return methods.ifdd_sarsa(domain)
    elif name == "kifdd-q":
        return methods.kifdd_q(domain)
    elif name == "kifdd-sarsa":
        return methods.kifdd_sarsa(domain)
    elif name == "rbfs-q":
        return methods.rbf_q(domain, seed=seed)
    elif name == "rbfs-sarsa":
        return methods.rbf_q(domain, seed=seed)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    run_experiment(
        select_domain,
        select_agent,
        default_max_steps=100000,
        default_num_policy_checks=30,
        default_checks_per_policy=1,
    )
