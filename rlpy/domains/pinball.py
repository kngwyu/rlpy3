"""Pinball domain for reinforcement learning
"""
import itertools

from itertools import product
from pathlib import Path

import numpy as np

try:
    from tkinter import Tk, Canvas
except ImportError:
    import warnings

    warnings.warn("TkInter is not found for Pinball.")

from rlpy.tools import __rlpy_location__
from .domain import Domain

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = [
    "Pierre-Luc Bacon",  # author of the original version
    "Austin Hays",  # adapted for RLPy and TKinter
]


class Pinball(Domain):
    """
    The goal of this domain is to maneuver a small ball on a plate into a hole.
    The plate may contain obstacles which should be avoided.

    **STATE:**
        The state is given by a 4-dimensional vector, consisting of position and
        velocity of the ball.

    **ACTIONS:**
        There are 5 actions, standing for slanting the  plat in x or y direction
        or a horizontal position
        of the plate.

    **REWARD:**
        Slanting the plate costs -4 reward in addition to -1 reward for each timestep.
        When the ball reaches the hole, the agent receives 10000 units of reward.

    **REFERENCE:**

    .. seealso::
       G.D. Konidaris and A.G. Barto:
       *Skill Discovery in Continuous Reinforcement Learning domains using Skill Chaining.*
       Advances in Neural Information Processing Systems 22, pages 1015-1023, December 2009.
    """

    #: default location of config files shipped with rlpy
    DEFAULT_CONFIG_DIR = Path(__rlpy_location__).joinpath("domains/PinballConfigs")

    @classmethod
    def default_cfg(cls, name="pinball_simple_single.json"):
        return cls.DEFAULT_CONFIG_DIR.joinpath(name)

    def __init__(
        self,
        noise=0.1,
        episode_cap=1000,
        config_file=DEFAULT_CONFIG_DIR.joinpath("pinball_simple_single.json"),
        xy_discr=10,
        v_discr=5,
        screen_width=500.0,
        screen_height=500.0,
    ):
        """
        :param config_file: Location of the configuration file.
        :param episode_cap: Maximum length of an episode
        :param noise: With probability noise, a uniformly random action is executed
        :param xy_discr: Number of x/y discritization for visualization
        :param c_discr: Number of v discritization for visualization
        """
        self.NOISE = noise
        self.actions = [
            PinballModel.ACC_X,
            PinballModel.DEC_Y,
            PinballModel.DEC_X,
            PinballModel.ACC_Y,
            PinballModel.ACC_NONE,
        ]
        super().__init__(
            num_actions=len(self.actions),
            statespace_limits=np.array(
                [[0.0, 1.0], [0.0, 1.0], [-2.0, 2.0], [-2.0, 2.0]]
            ),
            continuous_dims=[4],
            episode_cap=episode_cap,
        )
        self.environment = PinballModel(config_file, random_state=self.random_state)

        # Visualization stuffs
        self.screen = None
        self.heatmap = None
        self.xy_discr = xy_discr
        self.v_discr = v_discr
        self.screen_width = screen_width
        self.screen_height = screen_height

    def show_domain(self, _a=None):
        if self.screen is None:
            tk_window = Tk()
            tk_window.title("RLPy Pinball")
            width, height = self.screen_width, self.screen_height
            self.screen = Canvas(tk_window, width=width, height=height)
            self.screen.configure(background="LightGray")
            self.screen.pack()
            self.environment_view = PinballView(
                self.screen, width, height, self.environment,
            )
        self.environment_view.blit()
        self.screen.pack()
        self.screen.update()

    def step(self, a):
        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possible_actions())
        reward, is_terminal = self.environment.take_action(a)
        self.state = self.environment.get_state()
        return reward, self.state.copy(), is_terminal, self.possible_actions()

    def s0(self):
        self.environment.reset()
        self.state = self.environment.get_state()
        return self.state, self.is_terminal(), self.possible_actions()

    def possible_actions(self, s=0):
        return np.array(self.actions)

    def is_terminal(self):
        return self.environment.goal_reward() is not None

    def show_learning(self, representation):
        VMIN, VMAX = -100, 100
        if self.heatmap is None:
            self.heatmap = _PinballHeatMap(self.xy_discr, vmin=VMIN, vmax=VMAX)
        v_unit = 4.0 / self.v_discr
        xy_unit = 1.0 / self.xy_discr
        dat = np.zeros((self.xy_discr, self.xy_discr))
        for y_i, x_i in product(range(self.xy_discr), range(self.xy_discr)):
            x = xy_unit * x_i
            y = xy_unit * y_i
            q = 0
            for xdot_i, ydot_i in product(range(self.v_discr), range(self.v_discr)):
                xdot = -2.0 + xdot_i * v_unit
                ydot = -2.0 + ydot_i * v_unit
                s = np.array([x, y, xdot, ydot])
                q += representation.Qs(s, False).mean()
            dat[y_i, x_i] = q / (self.v_discr * self.v_discr)
        self.heatmap.update(dat)
        self.heatmap.draw()

    def close_visualizations(self):
        if self.screen is not None:
            self.screen.destroy()
        if self.heatmap is not None:
            self.heatmap.close()

    def all_states(self):
        s = []
        v_unit = 4.0 / self.v_discr
        xy_unit = 1.0 / self.xy_discr
        for y_i, x_i in product(range(self.xy_discr), range(self.xy_discr)):
            x = xy_unit * x_i
            y = xy_unit * y_i
            for xdot_i, ydot_i in product(range(self.v_discr), range(self.v_discr)):
                xdot = -2.0 + xdot_i * v_unit
                ydot = -2.0 + ydot_i * v_unit
                s.append([x, y, xdot, ydot])
        return np.stack(s)

    def change_start(self, new_start, index=0):
        self.environment.start_positions[index] = new_start

    def change_target(self, new_target, index=0):
        self.environment._targets.pos[index] = new_target


class _PinballHeatMap:
    def __init__(
        self,
        xy_discr,
        nrows=1,
        ncols=1,
        name="Pinball Value Function",
        cmap="ValueFunction-New",
        vmin=-10,
        vmax=10,
    ):
        from rlpy.tools.plotting import plt

        self.fig = plt.figure(name)
        cmap = plt.get_cmap(cmap)
        self.data_shape = xy_discr, xy_discr
        dummy_data = np.zeros(self.data_shape)
        self.imgs = []
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            img = ax.imshow(
                dummy_data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            cbar = ax.figure.colorbar(img, ax=ax)
            cbar.ax.set_ylabel("", rotation=-90, va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            self.imgs.append(img)

        self.fig.tight_layout()
        self.fig.canvas.draw()

        def close():
            plt.close(self.fig)

        self.close = close

    def update(self, data, index=0):
        self.imgs[index].set_data(data.reshape(self.data_shape))

    def draw(self):
        self.fig.canvas.draw()


class BallModel:

    """ This class maintains the state of the ball
    in the pinball domain. It takes care of moving
    it according to the current velocity and drag coefficient.

    """

    DRAG = 0.995
    VMAX = 2.0

    def __init__(self, start_position, radius):
        """
        :param start_position: The initial position
        :type start_position: np.ndarray
        :param radius: The ball radius
        :type radius: float
        """
        self.position = start_position
        self.radius = radius
        self.dxdy = np.zeros(2)

    def add_impulse(self, delta):
        """ Changes the momentum of the ball. """
        self.dxdy = np.clip(self.dxdy + delta / 5, -self.VMAX, self.VMAX)

    def add_drag(self):
        """ Add a fixed amount of drag to the current velocity. """
        self.dxdy *= self.DRAG

    def step(self):
        """ Moves the ball by one increment """
        self.position += self.dxdy * self.radius / 20.0


class PinballObstacle:

    """ This class represents a single polygon obstacle in the
    pinball domain and detects when a :class:`BallModel` hits it.

    When a collision is detected, it also provides a way to
    compute the appropriate effect to apply on the ball.
    """

    def __init__(self, points):
        """
        :param points: A list of points defining the polygon
        :type points: list of lists
        """
        self.points = np.array(points)
        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = max(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None

    def collision(self, ball):
        """ Determines if the ball hits this obstacle
        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        """

        self._double_collision = False
        x, y = ball.position
        if x + ball.radius < self.min_x or self.max_x < x - ball.radius:
            return False
        if y + ball.radius < self.min_y or self.max_y < y - ball.radius:
            return False

        intercept_found = False
        n_points = self.points.shape[0]
        for i in range(n_points):
            pt_pair = self.points[i], self.points[(i + 1) % n_points]
            if self._intercept_edge(*pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(pt_pair, self._intercept, ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        """
        Based of the collision detection result triggered in
        :func:`PinballObstacle.collision`, compute the change in velocity.

        :param ball: An instance of :class:`BallModel`
        :type ball: :class:`BallModel`
        """

        if self._double_collision:
            return -ball.dxdy

        # Normalize direction
        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        theta = angle_between(ball.dxdy, obstacle_vector) - np.pi
        intercept_theta = angle_between([-1.0, 0.0], obstacle_vector)
        theta += intercept_theta

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        velocity = np.linalg.norm(ball.dxdy)
        return np.array([np.cos(theta), np.sin(theta)]) * velocity

    def _select_edge(self, intersect1, intersect2, ball):
        """If the ball hits a corner, select one of two edges.
        :param intersect1: A pair of points defining an edge of the polygon
        :type intersect1: list of lists
        :param intersect2: A pair of points defining an edge of the polygon
        :type intersect2: list of lists
        :returns: The edge with the smallest angle with the velocity vector
        :rtype: list of lists
        """

        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = angle_between(ball.dxdy, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = angle_between(ball.dxdy, obstacle_vector2)
        if angle2 > np.pi:
            angle2 -= np.pi

        if np.abs(angle1 - np.pi / 2) < np.abs(angle2 - np.pi / 2):
            return intersect1
        return intersect2

    def _angle(self, v1, v2):
        """ Compute the angle difference between two vectors
        :param v1: The x,y coordinates of the vector
        :type: v1: list
        :param v2: The x,y coordinates of the vector
        :type: v2: list
        :rtype: float
        """

        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        return angle_diff

    def _intercept_edge(self, pt0, pt1, ball):
        """
        Compute the projection on and edge and find out if it intercept with the ball.
        """

        # Find the projection on an edge
        obstacle_edge = pt1 - pt0
        difference = np.array(ball.position) - pt0

        scalar_proj = np.clip(
            difference.dot(obstacle_edge) / obstacle_edge.dot(obstacle_edge), 0.0, 1.0
        )

        # Compute the distance to the closest point
        closest_pt = pt0 + obstacle_edge * scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius * ball.radius:
            # A collision only if the ball is not already moving away
            ball_to_obstacle = closest_pt - ball.position

            angle = angle_between(ball_to_obstacle, ball.dxdy)
            if angle > np.pi:
                angle = 2 * np.pi - angle

            if angle > np.pi / 1.99:
                return False

            return True
        else:
            return False


class PinballTarget:
    """
    Abstracts the goal position/radian of Pinball.
    """

    def __init__(
        self, target_pos, target_rad, target_color="red", target_reward_scale=1.0,
    ):
        if isinstance(target_pos[0], list):
            self.num_goals = len(target_pos)
        else:
            self.num_goals = 1
            target_pos = [target_pos]

        if isinstance(target_rad, float):
            target_rad = [target_rad] * self.num_goals
        if isinstance(target_reward_scale, float):
            target_reward_scale = [target_reward_scale] * self.num_goals
        if isinstance(target_color, str):
            target_color = [target_color] * self.num_goals

        self.pos = np.array(target_pos)
        self.rad = np.array(target_rad)
        self.reward_scale = np.array(target_reward_scale)
        self.color = target_color

    def __repr__(self):
        return f"Target(pos: {self.pos} rad: {self.rad} reward: {self.reward_scale})"

    def _collide(self, pos):
        for i in range(self.num_goals):
            dist = np.linalg.norm(np.array(pos) - np.array(self.pos[i]))
            if dist < self.rad[i]:
                return self.reward_scale[i]
        return None


class PinballModel:
    """
    This class is a self-contained model of the pinball domain for
    reinforcement learning.
    """

    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4

    STEP_PENALTY = -1
    THRUST_PENALTY = -5
    END_EPISODE = 10000

    ACTION_STEPS = 20

    class _Collision:
        def __init__(self, dxdy):
            self.dxdy = dxdy

    class _DoubleCollision:
        pass

    def __init__(self, config_file, random_state):
        """ Reads a configuration file for Pinball and draw the domain to screen
        """

        self.random_state = random_state
        self.action_effects = {
            self.ACC_X: np.array([1.0, 0.0]),
            self.ACC_Y: np.array([0.0, 1.0]),
            self.DEC_X: np.array([-1.0, 0.0]),
            self.DEC_Y: np.array([0.0, -1.0]),
            self.ACC_NONE: np.array([0.0, 0.0]),
        }
        import json

        # Set up the environment according to the configuration
        with config_file.open() as f:
            config = json.load(f)
        try:
            self.obstacles = list(map(PinballObstacle, config["obstacles"]))
            self._targets = PinballTarget(
                config["target_pos"],
                config["target_rad"],
                config.get("target_color", "red"),
                config.get("target_reward_scale", 1.0),
            )
            start_positions = np.array(config["start_pos"])
            ball_rad = config["ball_rad"]
        except KeyError as e:
            raise KeyError(f"Invalid Pinball config: missing a key {e}")

        self.start_positions = np.array(start_positions)
        self.ball = BallModel(self.sample_start(), ball_rad)

    def reset(self):
        self.ball.position = self.sample_start()
        self.ball.dxdy.fill(0.0)

    def sample_start(self):
        idx = self.random_state.randint(self.start_positions.shape[0])
        return self.start_positions[idx].copy()

    def get_state(self):
        """ Access the current 4-dimensional state vector.
        :returns: a list containing the x position, y position, xdot, ydot
        :rtype: np.ndarray
        """
        return np.concatenate((self.ball.position, self.ball.dxdy))

    def _detect_collision(self):
        dxdy = None
        for obs in self.obstacles:
            if obs.collision(self.ball):
                if dxdy is not None:
                    return PinballModel._DoubleCollision()
                dxdy = obs.collision_effect(self.ball)

        if dxdy is None:
            return None
        else:
            return PinballModel._Collision(dxdy)

    def take_action(self, action):
        """ Take a step in the environment

        :param action: The action to apply over the ball
        :type action: int

        """
        if isinstance(action, np.ndarray):
            action = action.item()
        self.ball.add_impulse(self.action_effects[action])

        for i in range(self.ACTION_STEPS):
            self.ball.step()
            # Detect collisions
            collision = self._detect_collision()

            if isinstance(collision, PinballModel._Collision):
                self.ball.dxdy = collision.dxdy
                if i + 1 == self.ACTION_STEPS:
                    self.ball.step()
            elif isinstance(collision, PinballModel._DoubleCollision):
                self.ball.dxdy *= -1

            reward_scale = self.goal_reward()
            if reward_scale is not None:
                return self.END_EPISODE * reward_scale, True

        self.ball.add_drag()
        self._check_bounds()

        if action == self.ACC_NONE:
            return self.STEP_PENALTY, False

        return self.THRUST_PENALTY, False

    def goal_reward(self):
        return self._targets._collide(self.ball.position)

    def _check_bounds(self):
        """ Make sure that the ball stays within the environment """
        for i in range(2):
            if self.ball.position[i] > 1.0:
                self.ball.position[i] = 0.95
            elif self.ball.position[i] < 0.0:
                self.ball.position[i] = 0.05

    def targets(self):
        return zip(self._targets.pos, self._targets.rad, self._targets.color)


class PinballView:
    """
    This class displays a :class:`PinballModel`.
    Used in conjunction with the :func:`run_pinballview` function,
    acting as a *controller*.
    """

    def __init__(self, screen, width, height, model):
        self.screen = screen
        self.width = width
        self.height = height
        self.model = model

        self.x, self.y = self._to_pixels(self.model.ball.position)
        self.rad = int(self.model.ball.radius * self.width)

        for obs in model.obstacles:
            coords_list = list(map(self._to_pixels, obs.points))
            chain = itertools.chain(*coords_list)
            coords = list(chain)
            self.screen.create_polygon(coords, fill="blue")
        self.screen.pack()

        for pos, rad, color in self.model.targets():
            x, y = self._to_pixels(pos)
            rad = int(rad * self.width)
            _ = self.drawcircle(self.screen, x, y, rad, color)
        self.ball_id = self.drawcircle(self.screen, self.x, self.y, self.rad, "black")
        self.screen.pack()

    def drawcircle(self, canv, x, y, rad, color):
        return canv.create_oval(x - rad, y - rad, x + rad, y + rad, width=0, fill=color)

    def _to_pixels(self, pt):
        """Converts from real units in the 0-1 range to pixel units

        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list
        """

        return [int(pt[0] * self.width), int(pt[1] * self.height)]

    def blit(self):
        """ Blit the ball onto the background surface """
        self.screen.coords(
            self.ball_id,
            self.x - self.rad,
            self.y - self.rad,
            self.x + self.rad,
            self.y + self.rad,
        )
        self.x, self.y = self._to_pixels(self.model.ball.position)
        self.screen.pack()


def angle_between(v1, v2):
    angle_diff = np.arctan2(*v1) - np.arctan2(*v2)
    if angle_diff < 0:
        angle_diff += 2 * np.pi
    return angle_diff


def run_pinballview(width, height, configuration):
    """
    Changed from original Pierre-Luc Bacon implementation to reflect
    the visualization changes in the PinballView Class.
    """

    width, height = float(width), float(height)
    tk_window = Tk()
    tk_window.title("RLPy Pinball")
    screen = Canvas(tk_window, width=500.0, height=500.0)
    screen.configure(background="LightGray")
    screen.pack()

    environment = PinballModel(configuration)
    environment_view = PinballView(screen, width, height, environment)

    actions = [
        PinballModel.ACC_X,
        PinballModel.DEC_Y,
        PinballModel.DEC_X,
        PinballModel.ACC_Y,
        PinballModel.ACC_NONE,
    ]
    done = False
    while not done:
        user_action = np.random.choice(actions)
        environment_view.blit()
        if environment.take_action(user_action)[1] == environment.END_EPISODE:
            done = True

        environment_view.blit()
        screen.update()
