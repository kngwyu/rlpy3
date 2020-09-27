"""Classic mountain car task."""
from rlpy.tools import bound, cm, from_a_to_b, lines, plt
from .domain import Domain
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = ["Josh Joseph", "Alborz Geramifard"]


class MountainCar(Domain):

    """
    The goal is to drive an under accelerated car up to the hill.\n

    **STATE:**        Position and velocity of the car [x, xdot] \n
    **ACTIONS:**      [Acc backwards, Coast, Acc forward] \n
    **TRANSITIONS:**  Move along the hill with some noise on the movement. \n
    **REWARD:**       -1 per step and 0 at or beyond goal (``x-goal > 0``). \n

    There is optional noise on vehicle acceleration.

    **REFERENCE:**
    Based on `RL-Community Java Implementation <http://library.rl-community.org/wiki/Mountain_Car_(Java)>`_
    """

    X_MIN = -1.2  #: Lower bound on domain position
    X_MAX = 0.6  #: Upper bound on domain position
    XDOT_MIN = -0.07  #: Lower bound on car velocity
    XDOT_MAX = 0.07  #: Upper bound on car velocity
    INIT_STATE = np.array([-0.5, 0.0])  #: Initial car state
    STEP_REWARD = -1  #: Penalty for each step taken before reaching the goal
    GOAL_REWARD = 0  #: Reward for reach the goal.
    GOAL = 0.5  #: X-Position of the goal location (Should be at/near hill peak)
    ACTIONS = [-1, 0, 1]  #: actions
    ACCEL_COEF = 0.001  #: Magnitude of acceleration action
    GRAVITY = -0.0025
    #: Hill peaks are generated as sinusoid; this is freq. of that sinusoid.
    HILL_PEAK_FREQ = 3.0

    # Used for visualization:
    X_DISCR = 20
    XDOT_DISCR = 20
    CAR_HEIGHT = 0.2
    CAR_WIDTH = 0.1
    ARROW_LENGTH = 0.2

    def __init__(self, noise=0, discount_factor=0.9, episode_cap=500):
        """
        :param noise: Magnitude of noise in stochastic velocity changes
        """
        super().__init__(
            num_actions=3,
            statespace_limits=np.array(
                [[self.X_MIN, self.X_MAX], [self.XDOT_MIN, self.XDOT_MAX]]
            ),
            continuous_dims=[0, 1],
            discount_factor=discount_factor,
            episode_cap=episode_cap,
        )
        self.noise = noise
        self.dim_names = ["X", "Xdot"]

        # Visualization stuffs
        self.vf_fig, self.vf_x, self.vf_xdot, self.vf_surf = (None,) * 4
        self.domain_fig, self.domain_ax = None, None
        self.policy_fig = None
        self.action_arrow = None
        if discount_factor < 1.0:
            self.min_return = (
                self.STEP_REWARD
                * (1.0 - discount_factor ** self.episode_cap)
                / (1.0 - discount_factor)
            )
        else:
            self.min_return = self.STEP_REWARD * self.episode_cap
        self.max_return = 0

    def step(self, a):
        """
        Take acceleration action *a*, adding noise as specified in ``__init__()``.

        """
        position, velocity = self.state
        noise = self.ACCEL_COEF * self.noise * 2 * (self.random_state.rand() - 0.5)
        velocity += (
            noise
            + self.ACTIONS[a] * self.ACCEL_COEF
            + np.cos(self.HILL_PEAK_FREQ * position) * self.GRAVITY
        )
        velocity = bound(velocity, self.XDOT_MIN, self.XDOT_MAX)
        position += velocity
        position = bound(position, self.X_MIN, self.X_MAX)
        if position <= self.X_MIN and velocity < 0:
            velocity = 0  # Bump into wall
        terminal = self.is_terminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        ns = np.array([position, velocity])
        self.state = ns.copy()
        return r, ns, terminal, self.possible_actions()

    def s0(self):
        self.state = self.INIT_STATE.copy()
        return self.state.copy(), self.is_terminal(), self.possible_actions()

    def is_terminal(self):
        """
        :return: ``True`` if the car has reached or exceeded the goal position.

        """

        return self.state[0] > self.GOAL

    def _init_domain_vis(self):
        self.domain_fig = plt.figure("MountainCar")
        self.domain_ax = self.domain_fig.add_subplot(111)
        # plot mountain
        mountain_x = np.linspace(self.X_MIN, self.X_MAX, 1000)
        mountain_y = np.sin(3 * mountain_x)
        self.domain_ax.fill_between(
            mountain_x, min(mountain_y) - self.CAR_HEIGHT * 2, mountain_y, color="g"
        )
        self.domain_ax.set_xlim([self.X_MIN - 0.2, self.X_MAX])
        self.domain_ax.set_ylim(
            [
                min(mountain_y) - self.CAR_HEIGHT * 2,
                max(mountain_y) + self.CAR_HEIGHT * 2,
            ]
        )
        # plot car
        self.car = lines.Line2D([], [], linewidth=20, color="b", alpha=0.8)
        self.domain_ax.add_line(self.car)
        # Goal
        self.domain_ax.plot(self.GOAL, np.sin(3 * self.GOAL), "yd", markersize=10.0)
        self.domain_ax.axis("off")
        self.domain_fig.show()

    def show_domain(self, a=0):
        """
         Plot the car and an arrow indicating the direction of accelaration
         Parts of this code was adopted from Jose Antonio Martin H.
         <jamartinh@fdi.ucm.es> online source code
        """
        pos, vel = self.state
        if self.domain_fig is None:
            self._init_domain_vis()
        car_middle_x = pos
        car_middle_y = np.sin(3 * pos)
        slope = np.arctan(3 * np.cos(3 * pos))
        car_back_x = car_middle_x - self.CAR_WIDTH * np.cos(slope) / 2.0
        car_front_x = car_middle_x + self.CAR_WIDTH * np.cos(slope) / 2.0
        car_back_y = car_middle_y - self.CAR_WIDTH * np.sin(slope) / 2.0
        car_front_y = car_middle_y + self.CAR_WIDTH * np.sin(slope) / 2.0
        self.car.set_data([car_back_x, car_front_x], [car_back_y, car_front_y])
        # Arrows
        if self.action_arrow is not None:
            self.action_arrow.remove()
            self.action_arrow = None

        if self.ACTIONS[a] > 0:
            self.action_arrow = from_a_to_b(
                car_front_x,
                car_front_y,
                car_front_x + self.ARROW_LENGTH * np.cos(slope),
                car_front_y + self.ARROW_LENGTH * np.sin(slope),
                "k",
                "arc3,rad=0",
                0,
                0,
                "simple",
            )
        elif self.ACTIONS[a] < 0:
            self.action_arrow = from_a_to_b(
                car_back_x,
                car_back_y,
                car_back_x - self.ARROW_LENGTH * np.cos(slope),
                car_back_y - self.ARROW_LENGTH * np.sin(slope),
                "r",
                "arc3,rad=0",
                0,
                0,
                "simple",
            )
        self.domain_fig.canvas.draw()
        self.domain_fig.canvas.flush_events()

    def _init_vf_vis(self):
        fig = plt.figure("Value Function")
        self.vf_ax = fig.add_subplot(111, projection="3d")
        x_space = np.linspace(self.X_MIN, self.X_MAX, self.X_DISCR)
        xdot_space = np.linspace(self.XDOT_MIN, self.XDOT_MAX, self.XDOT_DISCR)
        self.vf_x, self.vf_xdot = np.meshgrid(x_space, xdot_space)
        self.vf_ax.set_xlabel(r"$x$")
        self.vf_ax.set_ylabel(r"$\dot x$")
        return fig

    def show_learning(self, representation):
        pi = np.zeros((self.X_DISCR, self.XDOT_DISCR), np.uint8)
        V = np.zeros((self.X_DISCR, self.XDOT_DISCR))

        if self.vf_fig is None:
            self.vf_fig = self._init_vf_vis()

        for row, xDot in enumerate(
            np.linspace(self.XDOT_MIN, self.XDOT_MAX, self.XDOT_DISCR)
        ):
            for col, x in enumerate(np.linspace(self.X_MIN, self.X_MAX, self.X_DISCR)):
                s = np.array([x, xDot])
                Qs = representation.Qs(s, False)
                As = self.possible_actions()
                pi[row, col] = representation.best_action(s, False, As)
                V[col, row] = max(Qs)
        if self.vf_surf is not None:
            self.vf_surf.remove()
        self.vf_surf = self.vf_ax.plot_surface(
            self.vf_x, self.vf_xdot, V, cmap=cm.coolwarm, linewidth=0, antialiased=False,
        )
        self.vf_fig.canvas.draw()
