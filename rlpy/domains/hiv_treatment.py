"""HIV Treatment domain"""
from .domain import Domain
import numpy as np
from scipy.integrate import odeint
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


class HIVTreatment(Domain):

    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """

    state_names = ("T1", "T1*", "T2", "T2*", "V", "E")
    actions = np.array([[0.0, 0.0], [0.7, 0.0], [0.0, 0.3], [0.7, 0.3]])
    dt = 5  #: measurement every 5 days
    #: only update the graphs in show_domain every x steps
    show_domain_every = 20

    def __init__(self, logspace=True):
        """
        :params logspace: whether observed states are in log10 space or not
        """
        self.logspace = logspace
        if logspace:
            statespace_limits = np.array([[-5, 8]] * 6)
        else:
            statespace_limits = np.array([[0.0, 1e8]] * 6)
        super().__init__(
            num_actions=4,
            statespace_limits=statespace_limits,
            episode_cap=200,  #: total of 1000 days with a measurement every 5 days
            discount_factor=0.98,
            continuous_dims=np.arange(6),
        )
        # store samples of current episode for drawing
        self.episode_data = np.zeros((7, self.episode_cap + 1))

    def step(self, a):
        self.t += 1

        eps1, eps2 = self.actions[a]
        ns = odeint(dsdt, self.state, [0, self.dt], args=(eps1, eps2), mxstep=1000)[-1]
        T1, T2, T1s, T2s, V, E = ns
        # the reward function penalizes treatment because of side-effects
        reward = -0.1 * V - 2e4 * eps1 ** 2 - 2e3 * eps2 ** 2 + 1e3 * E
        self.state = ns.copy()
        if self.logspace:
            ns = np.log10(ns)

        self.episode_data[:-1, self.t] = self.state
        self.episode_data[-1, self.t - 1] = a
        return reward, ns, False, self.possible_actions()

    def possible_actions(self):
        return np.arange(4)

    def s0(self):
        self.t = 0
        self.episode_data[:] = np.nan
        # non-healthy stable state of the system
        s = np.array([163573.0, 5.0, 11945.0, 46.0, 63919.0, 24.0])
        self.state = s.copy()
        if self.logspace:
            return np.log10(s), self.is_terminal(), self.possible_actions()
        self.episode_data[:-1, 0] = s
        return s, self.is_terminal(), self.possible_actions()

    def show_domain(self, a=0, s=None):
        """
        shows a live graph of each concentration
        """
        # only update the graph every couple of steps, otherwise it is
        # extremely slow
        if self.t % self.show_domain_every != 0 and not self.t >= self.episode_cap:
            return

        n = self.state_space_dims + 1
        names = list(self.state_names) + ["Action"]
        colors = ["b", "b", "b", "b", "r", "g", "k"]
        handles = getattr(self, "_state_graph_handles", None)
        fig = plt.figure("HIVTreatment", figsize=(12, 10))
        if handles is None:
            handles = []
            f, axes = plt.subplots(n, sharex=True, num="HIVTreatment", figsize=(12, 10))
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


def dsdt(s, t, eps1, eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = 0.34
    k1 = 8e-7
    k2 = 1e-4
    delta = 0.7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.0
    c = 13.0
    rho1 = 1.0
    rho2 = 1.0
    lambdaE = 1
    bE = 0.3
    Kb = 100
    d_E = 0.25
    Kd = 500
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1.0 - eps1) * k1 * V * T1
    tmp2 = (1.0 - f * eps1) * k2 * V * T2
    dT1 = lambda1 - d1 * T1 - tmp1
    dT2 = lambda2 - d2 * T2 - tmp2
    dT1s = tmp1 - delta * T1s - m1 * E * T1s
    dT2s = tmp2 - delta * T2s - m2 * E * T2s
    dV = (
        (1.0 - eps2) * NT * delta * (T1s + T2s)
        - c * V
        - ((1.0 - eps1) * rho1 * k1 * T1 + (1.0 - f * eps1) * rho2 * k2 * T2) * V
    )
    dE = (
        lambdaE
        + bE * (T1s + T2s) / (T1s + T2s + Kb) * E
        - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E
        - deltaE * E
    )

    return np.array([dT1, dT2, dT1s, dT2s, dV, dE])


try:
    from .hiv_treatment_dynamics import dsdt
except Exception as e:
    print(e)
    print(
        "Cython extension for HIVTreatment dynamics not available, expect slow runtime"
    )
