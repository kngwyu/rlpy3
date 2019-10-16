"""Fourier representation"""
from .representation import Representation
from numpy.linalg import norm
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


class Fourier(Representation):
    """ Fourier representation.
    Represents the value function using a Fourier series of the specified
    order (eg 3rd order, 5th order, etc).
    See Konidaris, Osentoski, and Thomas, "Value Function Approximation in
    Reinforcement Learning using Fourier Basis" (2011).
    http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf
    """

    def __init__(self, domain, order=3, scaling=False):
        """
        :param domain: the problem :py:class:`~rlpy.domains.domain.Domain` to learn
        :param order: The degree of approximation to use in the Fourier series
            (eg 3rd order, 5th order, etc).  See reference paper in class API.

        """
        dims = domain.state_space_dims
        self.coeffs = np.indices((order,) * dims).reshape((dims, -1)).T
        super().__init__(domain, self.coeffs.shape[0])

        if scaling:
            coeff_norms = np.array(list(map(norm, self.coeffs)))
            coeff_norms[0] = 1.0
            self.alpha_scale = np.tile(1.0 / coeff_norms, (domain.actions_num,))
        else:
            self.alpha_scale = 1.0

    def phi_non_terminal(self, s):
        # normalize the state
        s_min, s_max = self.domain.statespace_limits.T
        norm_state = (s - s_min) / (s_max - s_min)
        return np.cos(np.pi * np.dot(self.coeffs, norm_state))

    def feature_type(self):
        return float

    def feature_learning_rate(self):
        return self.alpha_scale
