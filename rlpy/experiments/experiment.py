"""Standard Experiment for Learning Control in RL."""
import click
from collections import defaultdict
from copy import deepcopy
import json
import logging
import numpy as np
import os
import re
from rlpy.tools import (
    checkNCreateDirectory,
    clock,
    deltaT,
    hhmmss,
    plt,
    printClass,
    with_pdf_fonts,
    MARKERS,
)
from rlpy.tools.encoders import NpAwareEncoder
import rlpy.tools.results

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"


class Experiment(object):

    """
    The Experiment controls the training, testing, and evaluation of the
    agent. Reinforcement learning is based around
    the concept of training an :py:class:`~agents.agent.Agent` to solve a task,
    and later testing its ability to do so based on what it has learned.
    This cycle forms a loop that the experiment defines and controls. First
    the agent is repeatedly tasked with solving a problem determined by the
    :py:class:`~domains.Domain.Domain`, restarting after some termination
    condition is reached.
    (The sequence of steps between terminations is known as an *episode*.)

    Each time the Agent attempts to solve the task, it learns more about how
    to accomplish its goal. The experiment controls this loop of "training
    sessions", iterating over each step in which the Agent and Domain interact.
    After a set number of training sessions defined by the experiment, the
    agent's current policy is tested for its performance on the task.
    The experiment collects data on the agent's performance and then puts the
    agent through more training sessions. After a set number of loops, training
    sessions followed by an evaluation, the experiment is complete and the
    gathered data is printed and saved. For each section, training and
    evaluation, the experiment determines whether or not the visualization
    of the step should generated.

    The Experiment class is a base class that provides
    the basic framework for all RL experiments. It provides the methods and
    attributes that allow child classes to interact with the Agent
    and Domain classes within the RLPy library.

    .. note::
        All experiment implementations should inherit from this class.
    """

    #: The Main Random Seed used to generate other random seeds (we use a different seed for each experiment id)
    MAIN_SEED = 999999999
    #: Maximum number of runs used for averaging, specified so that enough
    #: random seeds are generated
    MAX_RUNS = 1000
    result = None

    log_template = "{total_steps: >6}: E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}"
    performance_log_template = "{total_steps: >6}: >>> E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}"

    def __init__(
        self,
        agent,
        domain,
        exp_id=1,
        max_steps=1000,
        config_logging=True,
        num_policy_checks=10,
        log_interval=1,
        path="Results/Temp",
        checks_per_policy=1,
        stat_bins_per_state_dim=0,
        capture_evaluation=False,
        **kwargs
    ):
        """
        :param agent: the :py:class:`~agents.agent.Agent` to use for learning the task.
        :param domain: the problem :py:class:`~domains.Domain.Domain` to learn
        :param exp_id: ID of this experiment (main seed used for calls to np.rand)
        :param max_steps: Total number of interactions (steps) before experiment termination.

        .. note::
            ``max_steps`` is distinct from ``episode_cap``; ``episode_cap`` defines the
            the largest number of interactions which can occur in a single
            episode / trajectory, while ``max_steps`` limits the sum of all
            interactions over all episodes which can occur in an experiment.

        :param num_policy_checks: Number of Performance Checks uniformly
            scattered along timesteps of the experiment
        :param log_interval: Number of seconds between log prints to console
        :param path: Path to the directory to be used for results storage
            (Results are stored in ``path/output_filename``)
        :param checks_per_policy: defines how many episodes should be run to
            estimate the performance of a single policy

        """
        self.exp_id = exp_id
        assert exp_id > 0
        self.agent = agent
        self.checks_per_policy = checks_per_policy
        self.domain = domain
        self.max_steps = max_steps
        self.num_policy_checks = num_policy_checks
        self.logger = logging.getLogger("rlpy.experiments.Experiment")
        self.log_interval = log_interval
        self.config_logging = config_logging
        self.path = path
        #: The name of the file used to store the data
        self.output_filename = ""
        # Array of random seeds. This is used to make sure all jobs start with
        # the same random seed
        self.random_seeds = np.random.RandomState(self.MAIN_SEED).randint(
            1, self.MAIN_SEED, self.MAX_RUNS
        )
        self.capture_evaluation = capture_evaluation
        self.result = defaultdict(list)
        if stat_bins_per_state_dim > 0:
            self.state_counts_learn = np.zeros(
                (domain.statespace_limits.shape[0], stat_bins_per_state_dim),
                dtype=np.long,
            )
            self.state_counts_perf = np.zeros(
                (domain.statespace_limits.shape[0], stat_bins_per_state_dim),
                dtype=np.long,
            )

    def _update_path(self, path):

        # compile and create output path
        self.full_path = self.compile_path(path)
        checkNCreateDirectory(self.full_path + "/")
        self.logger.info("Output:\t\t\t%s/%s" % (self.full_path, self.output_filename))
        # TODO set up logging to file for rlpy loggers
        self.log_filename = "{:0>3}.log".format(self.exp_id)
        if self.config_logging:
            rlpy_logger = logging.getLogger("rlpy")
            for h in rlpy_logger.handlers:
                rlpy_logger.removeHandler(h)
            rlpy_logger.addHandler(logging.StreamHandler())
            rlpy_logger.addHandler(
                logging.FileHandler(os.path.join(self.full_path, self.log_filename))
            )
            rlpy_logger.setLevel(logging.INFO)

    def seed_components(self):
        """
        set the initial seeds for all random number generators used during
        the experiment run based on the currently set ``exp_id``.
        """
        self._update_path(self.path)
        self.output_filename = "{:0>3}-results.json".format(self.exp_id)
        np.random.seed(self.random_seeds[self.exp_id - 1])
        self.domain.set_seed(self.random_seeds[self.exp_id - 1])
        # make sure the performance_domain has a different seed
        self.performance_domain.set_seed(self.random_seeds[self.exp_id + 20])

        # Its ok if use same seed as domain, random calls completely different
        self.agent.set_seed(self.random_seeds[self.exp_id - 1])

        self.log_filename = "{:0>3}.log".format(self.exp_id)
        if self.config_logging:
            rlpy_logger = logging.getLogger("rlpy")
            for h in rlpy_logger.handlers:
                if isinstance(h, logging.FileHandler):
                    rlpy_logger.removeHandler(h)
            rlpy_logger.addHandler(
                logging.FileHandler(os.path.join(self.full_path, self.log_filename))
            )

    def performance_run(self, total_steps, visualize=False):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        """

        # Set Exploration to zero and sample one episode from the domain
        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.agent.policy.turnOffExploration()

        s, eps_term, p_actions = self.performance_domain.s0()

        while not eps_term and eps_length < self.domain.episode_cap:
            a = self.agent.policy.pi(s, eps_term, p_actions)
            if visualize:
                self.performance_domain.show_domain(a)
                if self.capture_evaluation:
                    click.pause("Get ready to capture the window?")
                    self.capture_evaluation = False

            r, ns, eps_term, p_actions = self.performance_domain.step(a)
            self._gather_transition_statistics(s, a, ns, r, learning=False)
            s = ns
            eps_return += r
            eps_discount_return += (
                self.performance_domain.discount_factor ** eps_length * r
            )
            eps_length += 1
        if visualize:
            self.performance_domain.show_domain(a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain)
        # that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying
        # to accomodate them as an MDP

        return eps_return, eps_length, eps_term, eps_discount_return

    def printAll(self):
        """
        prints all information about the experiment
        """
        printClass(self)

    def _gather_transition_statistics(self, s, a, sn, r, learning=False):
        """
        This function can be used in subclasses to collect statistics
        about the transitions
        """
        if hasattr(self, "state_counts_learn") and learning:
            counts = self.state_counts_learn
        elif hasattr(self, "state_counts_perf") and not learning:
            counts = self.state_counts_perf
        else:
            return
        rng = self.domain.statespace_width
        d = counts.shape[-1] - 2
        s_norm = s - self.domain.statespace_limits[:, 0]
        idx = np.floor(s_norm / rng * d).astype("int")
        idx += 1
        idx[idx < 0] = 0
        idx[idx >= d + 2] = d + 1
        # import ipdb; ipdb.set_trace()
        counts[list(range(counts.shape[0])), idx] += 1

    def run(
        self, visualize_performance=0, visualize_learning=False, visualize_steps=False
    ):
        """
        Run the experiment and collect statistics / generate the results

        :param visualize_performance: (int)
            determines whether a visualization of the steps taken in
            performance runs are shown. 0 means no visualization is shown.
            A value n > 0 means that only the first n performance runs for a
            specific policy are shown (i.e., for n < checks_per_policy, not all
            performance runs are shown)
        :param visualize_learning: (boolean)
            show some visualization of the learning status before each
            performance evaluation (e.g. Value function)
        :param visualize_steps: (boolean)
            visualize all steps taken during learning
        """
        self.performance_domain = deepcopy(self.domain)
        self.performance_domain.performance = True
        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id
        total_steps = 0
        eps_steps = 0
        eps_return = 0
        episode_number = 0

        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.show_learning(self.agent.representation)

        # Used to bound the number of logs in the file
        start_log_time = clock()
        # Used to show the total time took the process
        self.start_time = clock()
        self.elapsed_time = 0
        # do a first evaluation to get the quality of the inital policy
        self.evaluate(total_steps, episode_number, visualize_performance)
        self.total_eval_time = 0.0
        terminal = True
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episode_cap:
                s, terminal, p_actions = self.domain.s0()
                a = self.agent.policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.agent.representation)

                # Output the current status if certain amount of time has been
                # passed
                eps_return = 0
                eps_steps = 0
                episode_number += 1
            # Act,Step
            r, ns, terminal, np_actions = self.domain.step(a)

            self._gather_transition_statistics(s, a, ns, r, learning=True)
            na = self.agent.policy.pi(ns, terminal, np_actions)

            total_steps += 1
            eps_steps += 1
            eps_return += r

            # Print Current performance
            if (terminal or eps_steps == self.domain.episode_cap) and deltaT(
                start_log_time
            ) > self.log_interval:
                start_log_time = clock()
                elapsedTime = deltaT(self.start_time)
                self.logger.info(
                    self.log_template.format(
                        total_steps=total_steps,
                        elapsed=hhmmss(elapsedTime),
                        remaining=hhmmss(
                            elapsedTime * (self.max_steps - total_steps) / total_steps
                        ),
                        totreturn=eps_return,
                        steps=eps_steps,
                        num_feat=self.agent.representation.features_num,
                    )
                )

            # learning
            self.agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
            s, a, p_actions = ns, na, np_actions
            # Visual
            if visualize_steps:
                self.domain.show(a, self.agent.representation)

            # Check Performance
            if total_steps % (self.max_steps // self.num_policy_checks) == 0:
                self.elapsed_time = deltaT(self.start_time) - self.total_eval_time

                # show policy or value function
                if visualize_learning:
                    self.domain.show_learning(self.agent.representation)

                self.evaluate(total_steps, episode_number, visualize_performance)
                self.total_eval_time += (
                    deltaT(self.start_time) - self.elapsed_time - self.total_eval_time
                )
                start_log_time = clock()

        # Visual
        if visualize_steps:
            self.domain.show(a, self.agent.representation)
        self.logger.info(
            "Total Experiment Duration %s" % (hhmmss(deltaT(self.start_time)))
        )

    def evaluate(self, total_steps, episode_number, visualize=0):
        """
        Evaluate the current agent within an experiment

        :param total_steps: (int)
                     number of steps used in learning so far
        :param episode_number: (int)
                        number of episodes used in learning so far
        """
        random_state = np.random.get_state()
        # random_state_domain = copy(self.domain.random_state)
        elapsedTime = deltaT(self.start_time)
        performance_return = 0.0
        performance_steps = 0.0
        performance_term = 0.0
        performance_discounted_return = 0.0
        for j in range(self.checks_per_policy):
            p_ret, p_step, p_term, p_dret = self.performance_run(
                total_steps, visualize=visualize > j
            )
            performance_return += p_ret
            performance_steps += p_step
            performance_term += p_term
            performance_discounted_return += p_dret
        performance_return /= self.checks_per_policy
        performance_steps /= self.checks_per_policy
        performance_term /= self.checks_per_policy
        performance_discounted_return /= self.checks_per_policy
        self.result["learning_steps"].append(total_steps)
        self.result["return"].append(performance_return)
        self.result["learning_time"].append(self.elapsed_time)
        self.result["num_features"].append(self.agent.representation.features_num)
        self.result["steps"].append(performance_steps)
        self.result["terminated"].append(performance_term)
        self.result["learning_episode"].append(episode_number)
        self.result["discounted_return"].append(performance_discounted_return)
        # reset start time such that performanceRuns don't count
        self.start_time = clock() - elapsedTime
        if total_steps > 0:
            remaining = hhmmss(
                elapsedTime * (self.max_steps - total_steps) / total_steps
            )
        else:
            remaining = "?"
        self.logger.info(
            self.performance_log_template.format(
                total_steps=total_steps,
                elapsed=hhmmss(elapsedTime),
                remaining=remaining,
                totreturn=performance_return,
                steps=performance_steps,
                num_feat=self.agent.representation.features_num,
            )
        )

        np.random.set_state(random_state)
        # self.domain.rand_state = random_state_domain

    def save(self):
        """Saves the experimental results to the ``results.json`` file
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        with open(results_fn, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True, cls=NpAwareEncoder)

    def load(self):
        """loads the experimental results from the ``results.txt`` file
        If the results could not be found, the function returns ``None``
        and the results array otherwise.
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        self.results = rlpy.tools.results.load_single(results_fn)
        return self.results

    def _plot_impl(self, y="return", x="learning_steps", save=False, show=True):
        labels = rlpy.tools.results.default_labels
        performance_fig = plt.figure("Performance")
        res = self.result
        plt.plot(res[x], res[y], lw=2, markersize=4, marker=MARKERS[0])
        plt.xlim(0, res[x][-1] * 1.01)
        y_arr = np.array(res[y])
        m = y_arr.min()
        M = y_arr.max()
        delta = M - m
        if delta > 0:
            plt.ylim(m - 0.1 * delta - 0.1, M + 0.1 * delta + 0.1)
        xlabel = labels[x] if x in labels else x
        ylabel = labels[y] if y in labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            path = os.path.join(
                self.full_path, "{:03}-performance.pdf".format(self.exp_id)
            )
            performance_fig.savefig(path, transparent=True, pad_inches=0.1)
        if show:
            plt.ioff()
            plt.show()

    def plot(self, y="return", x="learning_steps", save=False, show=True):
        """Plots the performance of the experiment
        This function has only limited capabilities.
        For more advanced plotting of results consider
        :py:class:`tools.Merger.Merger`.
        """
        with with_pdf_fonts():
            self._plot_impl(y, x, save, show)

    def compile_path(self, path):
        """
        An experiment path can be specified with placeholders. For
        example, ``Results/Temp/{domain}/{agent}/{representation}``.
        This functions replaces the placeholders with actual values.
        """
        variables = re.findall("{([^}]*)}", path)
        replacements = {}
        for v in variables:
            if v.lower().startswith("representation") or v.lower().startswith("policy"):
                obj = "self.agent." + v
            else:
                obj = "self." + v

            if obj.lower() in [
                "self.domain",
                "self.agent",
                "self.agent.policy",
                "self.agent.representation",
            ]:
                replacements[v] = "className({})".format(obj)
            else:
                replacements[v] = str(v)

        return path.format(**replacements)
