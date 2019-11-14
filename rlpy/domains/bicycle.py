"""Bicycle balancing task."""
from .domain import Domain
import numpy as np
from itertools import product
from rlpy.tools import plt

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class BicycleBalancing(Domain):

    """
    Simulation of balancing a bicycle.

    **STATE:**
    The state contains of 7 variables, 5 of which are observable.

    * ``omega:``     angle from the vertical to the bicycle [rad]
    * ``omega dot:`` angular velocity for omega [rad / s]
    * ``theta:``     angle the handlebars are displaced from normal [rad]
    * ``theta dot:`` angular velocity for theta [rad / s]
    * ``psi:``       angle formed by bicycle frame and x-axis [rad]

    [x_b:       x-coordinate where the back tire touches the ground [m]]
    [y_b:       y-coordinate where the back tire touches the ground [m]]

    *The state variables* x_b *and* y_b *are not observable.*

    **ACTIONS:**

    * T in {-2, 0, 2}:      the torque applied to the handlebar
    * d in {-.02, 0, .02}:  displacement of the rider

    i.e., 9 actions in total.


    **REFERENCE:**

    .. seealso::
        Ernst, D., Geurts, P. & Wehenkel, L. Tree-Based Batch Mode Reinforcement Learning.
        Journal of Machine Learning Research (2005) Issue 6

    .. warning::
        This domain is tested only marginally, use with a care.
    """

    state_names = (
        r"$\omega$",
        r"$\dot{\omega}$",
        r"$\theta$",
        r"$\dot{\theta}$",
        r"$\psi",
    )
    #: only update the graphs in show_domain every x steps
    show_domain_every = 20
    dt = 0.01  #: Frequency is ``1 / dt``.

    def __init__(self):
        self.actions = np.array(list(product([-2, 0, 2], [-0.02, 0.0, 0.02])))
        smax = np.array([np.pi * 12 / 180, np.pi, np.pi * 80 / 180, np.pi, np.pi])
        super().__init__(
            actions_num=self.actions.shape[0],
            statespace_limits=np.stack((-smax, smax), axis=1),
            discount_factor=0.98,
            continuous_dims=np.arange(5),
            episode_cap=50000,
        )
        self.episode_data = np.zeros((6, self.episode_cap + 1))
        self._state_graph_handles = None

    def step(self, a):
        self.t += 1
        s = self.state
        T, d = self.actions[a]
        omega, domega, theta, dtheta, psi = s

        v = 10 / 3.6
        g = 9.82
        d_CM = 0.3
        c = 0.66
        h = 0.94
        M_c = 15.0
        M_d = 1.7
        M_p = 60.0
        M = M_c + M_p
        r = 0.34
        dsigma = v / r
        I = 13 / 3.0 * M_c * h ** 2 + M_p * (h + d_CM) ** 2
        I_dc = M_d * r ** 2
        I_dv = 3 / 2.0 * M_d * r ** 2
        I_dl = M_d / 2 * r ** 2
        l = 1.11

        w = self.random_state.uniform(-0.02, 0.02)

        phi = omega + np.arctan(d + w) / h
        invr_f = np.abs(np.sin(theta)) / l
        invr_b = np.abs(np.tan(theta)) / l
        invr_CM = ((l - c) ** 2 + invr_b ** (-2)) ** (-0.5) if theta != 0.0 else 0.0

        nomega = omega + self.dt * domega
        ndomega = (
            domega
            + self.dt
            * (
                M * h * g * np.sin(phi)
                - np.cos(phi)
                * (
                    I_dc * dsigma * dtheta
                    + np.sign(theta)
                    * v ** 2
                    * (M_d * r * (invr_f + invr_b) + M * h * invr_CM)
                )
            )
            / I
        )
        out = theta + self.dt * dtheta
        rad80 = (80 / 180) * np.pi
        ntheta = out if abs(out) <= rad80 else np.sign(out) * rad80
        ndtheta = (
            dtheta + self.dt * (T - I_dv * dsigma * domega) / I_dl
            if abs(out) <= rad80
            else 0.0
        )
        npsi = psi + self.dt * np.sign(theta) * v * invr_b

        # Where are these three lines from? Having a hard time finding them in
        # the paper referenced
        npsi = npsi % (2 * np.pi)
        if npsi > np.pi:
            npsi -= 2 * np.pi

        ns = np.array([nomega, ndomega, ntheta, ndtheta, npsi])
        self.state = ns

        self.episode_data[:-1, self.t] = self.state
        self.episode_data[-1, self.t - 1] = a

        return self._reward(s), ns, self.is_terminal(), self.possible_actions()

    def is_terminal(self):
        s = self.state
        omega = s[0]
        return omega < -np.pi * 12.0 / 180 or omega > np.pi * 12.0 / 180.0

    def _reward(self, s):
        return -1.0 if self.is_terminal() else 0.0

    def possible_actions(self):
        return np.arange(9)

    def s0(self):
        # non-healthy stable state of the system
        self.t = 0
        s = np.zeros(5)
        self.state = s
        self.episode_data[:] = np.nan
        self.episode_data[:-1, 0] = s
        return s, self.is_terminal(), self.possible_actions()

    def show_domain(self, a=0, s=None):
        """
        shows a live graph of each observable dimension
        """
        # only update the graph every couple of steps, otherwise it is
        # extremely slow
        if self.t % self.show_domain_every != 0 and not self.t >= self.episode_cap:
            return

        n = self.state_space_dims + 1
        names = list(self.state_names) + ["Action"]
        colors = ["m", "c", "b", "r", "g", "k"]
        handles = self._state_graph_handles
        fig = plt.figure("Bicycle", figsize=(12, 10))
        if handles is None:
            handles = []
            f, axes = plt.subplots(n, sharex=True, num="Bicycle", figsize=(12, 10))
            f.subplots_adjust(hspace=0.1)
            for i in range(n):
                ax = axes[i]
                d = np.arange(self.episode_cap + 1) * 5
                ax.set_ylabel(names[i])
                ax.locator_params(tight=True, nbins=4)
                handles.append(ax.plot(d, self.episode_data[i], color=colors[i])[0])
            self._state_graph_handles = handles
            ax.set_xlabel("Days")
        for i in range(n):
            handles[i].set_ydata(self.episode_data[i])
            ax = handles[i].axes
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()


class BicycleRiding(BicycleBalancing):
    def _reward(self, s):
        ns = self.state
        psi = s[-1]
        npsi = ns[-1]
        if self.is_terminal():
            return -1.0
        return 0.1 * (psi - npsi)
