from rlpy.Domains import GridWorld
from rlpy.Representations import Tabular
from scipy.optimize import check_grad, approx_fprime
from rlpy.Policies.gibbs import GibbsPolicy
import numpy as np


def test_fdcheck_dlogpi():
    domain = GridWorld()
    representation = Tabular(domain=domain, discretization=20)
    policy = GibbsPolicy(representation=representation)

    def f(wv, s, a):
        policy.representation.weight_vec = wv
        return np.log(policy.prob(s, a))

    def df(wv, s, a):
        policy.representation.weight_vec = wv
        return policy.dlogpi(s, a)

    def df_approx(wv, s, a):
        return approx_fprime(wv, f, 1e-10, s, a)

    wvs = np.random.rand(10, len(representation.weight_vec))
    for i in range(10):
        s = np.array([np.random.randint(4), np.random.randint(5)])
        a = np.random.choice(domain.possibleActions(s))
        for wv in wvs:
            error = check_grad(f, df, wv, s, a)
            assert np.abs(error) < 1e-6, 'Error={}'.format(error)
