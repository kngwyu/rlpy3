"""Helicopter hovering task."""
from .domain import Domain
import numpy as np
import rlpy.tools.transformations as trans
from rlpy.tools.general_tools import cartesian
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from mpl_toolkits.mplot3d import proj3d

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"


class Arrow3D(FancyArrowPatch):

    """
    Helper class for plotting arrows in 3d
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class HelicopterHoverExtended(Domain):

    """
    Implementation of a simulator that models one of the Stanford
    autonomous helicopters (an XCell Tempest helicopter) in the flight
    regime close to hover.

    Adapted from the
    `RL-Community Java Implementation <http://library.rl-community.org/wiki/Helicopter_(Java)>`_

    **STATE:**
    The state of the helicopter is described by a 20-dimensional vector
    with the following entries:

    * 0: xerr [helicopter x-coord position - desired x-coord position] -- helicopter's x-axis points forward
    * 1: yerr [helicopter y-coord position - desired y-coord position] -- helicopter's y-axis points to the right
    * 2: zerr [helicopter z-coord position - desired z-coord position] -- helicopter's z-axis points down
    * 3: u [forward velocity]
    * 4: v [sideways velocity (to the right)]
    * 5: w [downward velocity]
    * 6: p [angular rate around helicopter's x axis]
    * 7: q [angular rate around helicopter's y axis]
    * 8: r [angular rate around helicopter's z axis]
    * 9-12: orientation of heli in world as quaterion
    * 13-18: current noise due to gusts (usually not observable!)
    * 19: t number of timesteps in current episode

    **REFERENCE:**

    .. seealso::
        Abbeel, P., Ganapathi, V. & Ng, A. Learning vehicular dynamics,
        with application to modeling helicopters.
        Advances in Neural Information Systems (2006).

    """

    MAX_POS = 20.0  #: [m]  maximum deviation in position in each dimension
    MAX_VEL = 10.0  #: [m/s] maximum velocity in each dimension
    MAX_ANG_RATE = 4 * np.pi  # : maximum angular velocity
    MAX_ANG = 1.0
    WIND_MAX = 5.0  # : maximum gust indensity
    MIN_QW_BEFORE_HITTING_TERMINAL_STATE = np.cos(30.0 / 2.0 * np.pi / 180.0)

    wind = np.array([0.0, 0.0, 0.0])  #: wind in neutral orientation
    discount_factor = 0.95  #: discount factor
    gust_memory = 0.8
    domain_fig = None

    episode_cap = 6000
    # model specific parameters from the learned model
    noise_std = np.array([0.1941, 0.2975, 0.6058, 0.1508, 0.2492, 0.0734])
    drag_vel_body = np.array([0.18, 0.43, 0.49])
    drag_ang_rate = np.array([12.78, 10.12, 8.16])
    u_coeffs = np.array([33.04, -33.32, 70.54, -42.15])
    tail_rotor_side_thrust = -0.54

    dt = 0.01  #: length of one timestep
    continuous_dims = np.arange(20)

    # create all combinations of possible actions
    _action_bounds = np.array([[-2.0, 2.0]] * 4)
    # maximum action: 2
    _actions_dim = np.array([[-0.2, -0.05, 0.05, 0.2]] * 3 + [[0.0, 0.15, 0.3, 0.5]])
    actions = cartesian(list(_actions_dim))  #: all possible actions

    def __init__(
        self,
        statespace_limits=None,
        continuous_dims=np.arange(20),
        noise_level=1.0,
        discount_factor=0.95,
        episode_cap=6000,
    ):
        self.statespace_limits_full = np.array(
            [[-self.MAX_POS, self.MAX_POS]] * 3
            + [[-self.MAX_VEL, self.MAX_VEL]] * 3
            + [[-self.MAX_ANG_RATE, self.MAX_ANG_RATE]] * 3
            + [[-self.MAX_ANG, self.MAX_ANG]] * 4
            + [[-2.0, 2.0]] * 6
            + [[0, episode_cap]]
        )
        self.noise_level = noise_level
        if statespace_limits is None:
            statespace_limits = self.statespace_limits_full
        super().__init__(
            actions_num=np.prod(self.actions.shape[0]),
            statespace_limits=statespace_limits,
            discount_factor=discount_factor,
            episode_cap=episode_cap,
        )

    def s0(self):
        self.state = np.zeros((20))
        self.state[9] = 1.0
        return self.state.copy(), self.is_terminal(), self.possible_actions()

    def is_terminal(self):
        s = self.state
        if np.any(self.statespace_limits_full[:9, 0] > s[:9]) or np.any(
            self.statespace_limits_full[:9, 1] < s[:9]
        ):
            return True

        if len(s) <= 12:
            w = np.sqrt(1.0 - np.sum(s[9:12] ** 2))
        else:
            w = s[9]

        return np.abs(w) < self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE

    def _get_reward(self):
        s = self.state
        if self.is_terminal():
            r = -np.sum(self.statespace_limits[:9, 1] ** 2)
            # r -= np.sum(self.statespace_limits[10:12, 1] ** 2)
            r -= 1.0 - self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE ** 2
            return r * (self.episode_cap - s[-1])
        else:
            return -np.sum(s[:9] ** 2) - np.sum(s[10:12] ** 2)

    def possible_actions(self, s=None):
        return np.arange(self.actions_num)

    def step(self, a):
        a = self.actions[a]
        # make sure the actions are not beyond their limits
        a = np.maximum(
            self._action_bounds[:, 0], np.minimum(a, self._action_bounds[:, 1])
        )

        pos, vel, ang_rate, ori_bases, q = self._state_in_world(self.state)
        t = self.state[-1]
        gust_noise = self.state[13:19]
        gust_noise = (
            self.gust_memory * gust_noise
            + (1.0 - self.gust_memory)
            * self.random_state.randn(6)
            * self.noise_level
            * self.noise_std
        )
        # update noise which simulates gusts
        for i in range(10):
            # Euler integration
            # position
            pos += self.dt * vel
            # compute acceleration on the helicopter
            vel_body = self._in_world_coord(vel, q)
            wind_body = self._in_world_coord(self.wind, q)
            wind_body[-1] = 0.0  # the java implementation
            # has it this way
            acc_body = -self.drag_vel_body * (vel_body + wind_body)
            acc_body[-1] += self.u_coeffs[-1] * a[-1]
            acc_body[1] += self.tail_rotor_side_thrust
            acc_body += gust_noise[:3]
            acc = self._in_body_coord(acc_body, q)
            acc[-1] += 9.81  # gravity

            # velocity
            vel += self.dt * acc
            # orientation
            tmp = self.dt * ang_rate
            qdt = trans.quaternion_about_axis(np.linalg.norm(tmp), tmp)
            q = trans.quaternion_multiply(q, qdt)
            # assert np.allclose(1., np.sum(q**2))
            # angular accelerations
            ang_acc = -ang_rate * self.drag_ang_rate + self.u_coeffs[:3] * a[:3]
            ang_acc += gust_noise[3:]

            ang_rate += self.dt * ang_acc

        st = np.zeros_like(self.state)
        st[:3] = -self._in_body_coord(pos, q)
        st[3:6] = self._in_body_coord(vel, q)
        st[6:9] = ang_rate
        st[9:13] = q
        st[13:19] = gust_noise
        st[-1] = t + 1
        self.state = st.copy()
        return (self._get_reward(), st, self.is_terminal(), self.possible_actions())

    def _state_in_world(self, s):
        """
        transforms state from body coordinates in world coordinates
        .. warning::

            angular rate still in body frame!

        """
        pos_body = s[:3]
        vel_body = s[3:6]
        ang_rate = s[6:9].copy()
        q = s[9:13].copy()

        pos = self._in_world_coord(-pos_body, q)
        vel = self._in_world_coord(vel_body, q)

        rot = trans.quaternion_matrix(trans.quaternion_conjugate(q))[:3, :3]
        return pos, vel, ang_rate, rot, q

    def _in_body_coord(self, p, q):
        """
        q is the inverse quaternion of the rotation of the helicopter in world coordinates
        """
        q_pos = np.zeros((4))
        q_pos[1:] = p
        q_p = trans.quaternion_multiply(
            trans.quaternion_multiply(q, q_pos), trans.quaternion_conjugate(q)
        )
        return q_p[1:]

    def _in_world_coord(self, p, q):
        """
        q is the inverse quaternion of the rotation of the helicopter in world coordinates
        """
        return self._in_body_coord(p, trans.quaternion_conjugate(q))

    def show_domain(self, a=None):
        s = self.state
        if a is not None:
            a = self.actions[a].copy() * 3  # amplify for visualization
        pos, vel, ang_rate, ori_bases, _ = self._state_in_world(s)
        coords = np.zeros((3, 3, 2)) + pos[None, :, None]
        coords[:, :, 1] += ori_bases * 4
        u, v = np.mgrid[0 : 2 * np.pi : 10j, 0:2:1]

        # rotor coordinates
        coord = np.zeros([3] + list(u.shape))
        coord[0] = 0.1 * np.sin(u) * v
        coord[1] = 0.0
        coord[2] = 0.1 * np.cos(u) * v
        coord[0] -= 0.8
        coord_side = np.einsum("ij,jkl->ikl", np.linalg.pinv(ori_bases), coord)
        coord_side += pos[:, None, None]

        coord = np.zeros([3] + list(u.shape))
        coord[0] = 0.6 * np.cos(u) * v
        coord[1] = 0.6 * np.sin(u) * v
        coord[2] = -0.4
        coord_main = np.einsum("ij,jkl->ikl", np.linalg.pinv(ori_bases), coord)
        coord_main += pos[:, None, None]

        style = dict(fc="r", ec="r", lw=2.0, head_width=0.05, head_length=0.1)
        if self.domain_fig is None:
            self.domain_fig = plt.figure("HelicopterHover", figsize=(12, 8))
            # action axes
            ax1 = plt.subplot2grid((1, 3), (0, 0), frameon=False)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            lim = 2  # self.MAX_POS
            ax1.set_xlim(-lim, lim)
            ax1.set_ylim(-lim, lim)
            if a is None:
                a = np.zeros((4))
            # main rotor
            ax1.add_artist(Circle(np.zeros((2)), radius=0.6))
            ax1.add_artist(Ellipse(np.array([0, 1.5]), height=0.3, width=0.02))
            # TODO make sure the actions are plotted right
            # main rotor direction?
            arr1 = ax1.arrow(0, 0, a[0], 0, **style)
            arr2 = ax1.arrow(0, 0, 0, a[1], **style)
            # side rotor throttle?
            arr3 = ax1.arrow(0, 1.5, a[2], 0, **style)
            # main rotor throttle
            arr4 = ax1.arrow(1.5, 0, 0, a[3], **style)

            self.action_arrows = (arr1, arr2, arr3, arr4)
            self.action_ax = ax1
            self.action_ax.set_aspect("equal")

            ax = plt.subplot2grid((1, 3), (0, 1), colspan=2, projection="3d")
            ax.view_init(elev=np.pi)
            # print origin
            x = Arrow3D(
                [0, 2],
                [0, 0],
                [0, 0],
                mutation_scale=30,
                lw=1,
                arrowstyle="-|>",
                color="r",
            )
            y = Arrow3D(
                [0, 0],
                [0, 2],
                [0, 0],
                mutation_scale=30,
                lw=1,
                arrowstyle="-|>",
                color="b",
            )
            z = Arrow3D(
                [0, 0],
                [0, 0],
                [0, 2],
                mutation_scale=30,
                lw=1,
                arrowstyle="-|>",
                color="g",
            )
            ax.add_artist(x)
            ax.add_artist(y)
            ax.add_artist(z)

            # print helicopter coordinate axes
            x = Arrow3D(
                *coords[0], mutation_scale=30, lw=2, arrowstyle="-|>", color="r"
            )
            y = Arrow3D(
                *coords[1], mutation_scale=30, lw=2, arrowstyle="-|>", color="b"
            )
            z = Arrow3D(
                *coords[2], mutation_scale=30, lw=2, arrowstyle="-|>", color="g"
            )
            ax.add_artist(x)
            ax.add_artist(y)
            ax.add_artist(z)
            self.heli_arrows = (x, y, z)

            self._wframe_main = ax.plot_wireframe(
                coord_main[0], coord_main[1], coord_main[2], color="k"
            )
            self._wframe_side = ax.plot_wireframe(
                coord_side[0], coord_side[1], coord_side[2], color="k"
            )
            self._ax = ax
            lim = 5  # self.MAX_POS
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.view_init(elev=-135)
            plt.show()
        else:
            self.heli_arrows[0]._verts3d = tuple(coords[0])
            self.heli_arrows[1]._verts3d = tuple(coords[1])
            self.heli_arrows[2]._verts3d = tuple(coords[2])
            ax = self._ax
            ax.collections.remove(self._wframe_main)
            ax.collections.remove(self._wframe_side)
            for arr in self.action_arrows:
                self.action_ax.artists.remove(arr)
            ax1 = self.action_ax
            # TODO make sure the actions are plotted right
            # main rotor direction?
            arr1 = ax1.arrow(0, 0, a[0], 0, **style)
            arr2 = ax1.arrow(0, 0, 0, a[1], **style)
            # side rotor throttle?
            arr3 = ax1.arrow(0, 1.5, a[2], 0, **style)
            # main rotor throttle
            arr4 = ax1.arrow(1.5, 0, 0, a[3], **style)
            self.action_arrows = (arr1, arr2, arr3, arr4)

            self._wframe_main = ax.plot_wireframe(
                coord_main[0], coord_main[1], coord_main[2], color="k"
            )
            self._wframe_side = ax.plot_wireframe(
                coord_side[0], coord_side[1], coord_side[2], color="k"
            )
            lim = 5  # self.MAX_POS
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.view_init(elev=-135)
            self.domain_fig.canvas.draw()


class HelicopterHover(HelicopterHoverExtended):

    """
    .. warning::

        This domain has an internal hidden state, as it actually is
        a POMDP. Besides the 12-dimensional observable state, there is an internal
        state saved as ``self.hidden_state_`` (time and long-term noise which
        simulated gusts of wind).
        be aware of this state if you use this class to produce samples which are
        not in order

    Implementation of a simulator that models one of the Stanford
    autonomous helicopters (an XCell Tempest helicopter) in the flight
    regime close to hover.

    Adapted from the
    `RL-Community Java Implementation <http://library.rl-community.org/wiki/Helicopter_(Java)>`_

    **STATE:**
    The state of the helicopter is described by a 12-dimensional vector
    with the following entries:

    * 0: xerr [helicopter x-coord position - desired x-coord position] -- helicopter's x-axis points forward
    * 1: yerr [helicopter y-coord position - desired y-coord position] -- helicopter's y-axis points to the right
    * 2: zerr [helicopter z-coord position - desired z-coord position] -- helicopter's z-axis points down
    * 3: u [forward velocity]
    * 4: v [sideways velocity (to the right)]
    * 5: w [downward velocity]
    * 6: p [angular rate around helicopter's x axis]
    * 7: q [angular rate around helicopter's y axis]
    * 8: r [angular rate around helicopter's z axis]
    * 9-11: orientation of the world in the heli system as quaterion

    **REFERENCE:**

    .. seealso::
        Abbeel, P., Ganapathi, V. & Ng, A. Learning vehicular dynamics,
        with application to modeling helicopters.
        Advances in Neural Information Systems (2006).
    """

    episode_cap = 6000
    MAX_POS = 20.0  # m
    MAX_VEL = 10.0  # m/s
    MAX_ANG_RATE = 4 * np.pi
    MAX_ANG = 1.0
    WIND_MAX = 5.0

    def __init__(self):
        st_max = np.array(
            [self.MAX_POS] * 3
            + [self.MAX_VEL] * 3
            + [self.MAX_ANG_RATE] * 3
            + [self.MAX_ANG] * 3
        )
        statespace_limits = np.stack((-st_max, st_max), axis=1)
        super().__init__(
            statespace_limits=statespace_limits, continuous_dims=np.arange(12)
        )

    def s0(self):
        # self.hidden_state_ = np.zeros((8))
        # self.hidden_state_[0] = 1.
        s_full, term, p_actions = super(HelicopterHover, self).s0()
        s, _ = self._split_state(s_full)
        return s, term, p_actions

    def _split_state(self, s):
        s_observable = np.zeros((12))
        s_observable[:9] = s[:9]
        s_observable[9:12] = s[10:13]
        s_hidden = np.zeros((8))
        s_hidden[0] = s[9]
        s_hidden[1:] = s[13:]
        return s_observable, s_hidden

    def step(self, a):
        # s_extended = self._augment_state(s)
        r, st, term, p_actions = super(HelicopterHover, self).step(a)
        st, _ = self._split_state(st)
        return (r, st, term, p_actions)
