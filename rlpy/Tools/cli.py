import click
from rlpy.Experiments import Experiment


def get_experiment(
    domain,
    agent_selector,
    default_max_steps=1000,
    default_num_policy_checks=10,
    default_checks_per_policy=10,
    **kwargs
):
    @click.group()
    @click.option(
        "--agent", type=str, default=None, help="The name of agent you want to run"
    )
    @click.option("--seed", type=int, default=1, help="The problem to learn")
    @click.option(
        "--max-steps",
        type=int,
        default=default_max_steps,
        help="Total number of interactions",
    )
    @click.option(
        "--num-policy-checks",
        type=int,
        default=default_num_policy_checks,
        help="Total number of evaluation time",
    )
    @click.option(
        "--checks-per-policy",
        type=int,
        default=default_checks_per_policy,
        help="Number of evaluation per 1 evaluation time",
    )
    @click.option("--log-interval", type=int, default=10, help="Number of seconds")
    @click.option(
        "--log-dir",
        type=str,
        default="Results/Temp",
        help="The directory to be used for storing the logs",
    )
    @click.pass_context
    def experiment(
        ctx,
        agent,
        seed,
        max_steps,
        num_policy_checks,
        checks_per_policy,
        log_interval,
        log_dir,
    ):
        agent = agent_selector(agent, seed)
        ctx.obj["experiment"] = Experiment(
            agent,
            domain,
            exp_id=seed,
            max_steps=max_steps,
            num_policy_checks=num_policy_checks,
            checks_per_policy=checks_per_policy,
            log_interval=log_interval,
            log_dir=log_dir,
            **kwargs
        )

    @experiment.command(help="Train the agent")
    @click.option(
        "--visualize-performance",
        default=0,
        type=int,
        help="The number of visualization steps during performance runs",
    )
    @click.option(
        "--visualize-learning",
        is_flag=True,
        help="Visualize of the learning status before each evaluation",
    )
    @click.option(
        "--visualize-steps", is_flag=True, help="Visualize all steps during learning"
    )
    @click.option(
        "--plot-result", is_flag=True, help="Visualize the result"
    )
    @click.pass_context
    def train(ctx, visualize_performance, visualize_learning, visualize_steps, plot_result):
        exp = ctx.obj["experiment"]
        exp.run(visualize_performance, visualize_learning, visualize_steps)
        if plot_result:
            exp.plot()

    return experiment


def run_experiment(*args, **kwargs):
    get_experiment(*args, **kwargs)(obj={})
