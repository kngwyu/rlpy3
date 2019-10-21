"""
Representations which use local bases function (e.g. kernels) distributed
in the statespace according to some scheme (e.g. grid, random, on previous
samples)
"""
from .representation import Representation
import numpy as np
from rlpy.tools.general_tools import add_new_features
import matplotlib.pyplot as plt

try:
    from .kernels import batch
except ImportError:
    from .slow_kernels import batch
    import warnings

    warnings.warn("C-Extensions for kernels not available, expect slow runtime")

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"


class LocalBases(Representation):
    """
    abstract base class for representations that use local basis functions
    """

    def __init__(self, domain, kernel, normalization=False, seed=1, **kwargs):
        """
        :param domain: domain to learn on.
        :param kernel: function handle to use for kernel function evaluations.
        :param normalization: (Boolean) If true, normalize feature vector so
            that sum( phi(s) ) = 1.

        Associates a kernel function with each
        """
        super().__init__(domain, 0, seed=seed)
        self.kernel = batch[kernel.__name__]
        self.normalization = normalization
        #: centers of bases
        self.centers = np.zeros((0, domain.statespace_limits.shape[0]))
        #: widths of bases
        self.widths = np.zeros((0, domain.statespace_limits.shape[0]))

    def phi_non_terminal(self, s):
        v = self.kernel(s, self.centers, self.widths)
        if self.normalization and not v.sum() == 0.0:
            # normalize such that each vector has a l1 norm of 1
            v /= v.sum()
        return v

    def plot_2d_feature_centers(self, d1=None, d2=None):
        """
        :param d1: 1 (of 2 possible) indices of dimensions to plot; ignore all
            others, purely visual.
        :param d2: 1 (of 2 possible) indices of dimensions to plot; ignore all
            others, purely visual.

        Phe centers of all features in dimension d1 and d2.
        If no dimensions are specified, the first two continuous dimensions
        are shown.

        """
        if d1 is None and d2 is None:
            # just take the first two dimensions
            d1, d2 = self.domain.continuous_dims[:2]
        plt.figure("Feature Dimensions {} and {}".format(d1, d2))
        for i in range(self.centers.shape[0]):
            plt.plot([self.centers[i, d1]], [self.centers[i, d2]], "r", marker="x")
        plt.draw()

    def feature_type(self):
        return float


class NonparametricLocalBases(LocalBases):
    def __init__(self, domain, kernel, max_similarity=0.9, resolution=5, **kwargs):
        """
        :param domain: domain to learn on.
        :param kernel: function handle to use for kernel function evaluations.
        :param max_similarity: threshold to allow feature to be added to
            representation.  Larger max_similarity makes it \"easier\" to add
            more features by permitting larger values of phi(s) before
            discarding.  (An existing feature function in phi() with large value
            at phi(s) implies that it is very representative of the true
            function at *s*.  i.e., the value of a feature in phi(s) is
            inversely related to the \"similarity\" of a potential new feature.
        :param resolution: to be used by the ``kernel()`` function, see parent.
            Determines *width* of basis functions, eg sigma in Gaussian basis.
        """
        super().__init__(domain, kernel, **kwargs)
        self.max_similarity = max_similarity
        self.common_width = domain.statespace_width / resolution
        self.features_num = 0

    def pre_discover(self, s, terminal, a, sn, terminaln):
        norm = self.normalization
        expanded = 0
        self.normalization = False
        if not terminal:
            phi_s = self.phi_non_terminal(s)
            if np.all(phi_s < self.max_similarity):
                self._add_feature(s)
                expanded += 1
        if not terminaln:
            phi_s = self.phi_non_terminal(sn)
            if np.all(phi_s < self.max_similarity):
                self._add_feature(sn)
                expanded += 1
        self.normalization = norm
        return expanded

    def _add_feature(self, center):
        self.features_num += 1
        self.centers = np.vstack((self.centers, center))
        self.widths = np.vstack((self.widths, self.common_width))
        # TODO if normalized, use Q estimate for center to fill weight
        self.weight = add_new_features(self.weight)


class RandomLocalBases(LocalBases):
    def __init__(
        self,
        domain,
        kernel,
        num=100,
        resolution_min=5,
        resolution_max=None,
        seed=1,
        **kwargs
    ):
        """
        :param domain: domain to learn on.
        :param kernel: function handle to use for kernel function evaluations.
        :param num: Fixed number of feature (kernel) functions to use in
            EACH dimension. (for a total of features_num=numDims * num)
        :param resolution_min: resolution selected uniform random, lower bound.
        :param resolution_max: resolution selected uniform random, upper bound.
        :param seed: the random seed to use when scattering basis functions.

        Randomly scatter ``num`` feature functions throughout the domain, with
        sigma / noise parameter selected uniform random between
        ``resolution_min`` and ``resolution_max``.  NOTE these are
        sensitive to the choice of coordinate (scale with coordinate units).
        """
        super().__init__(domain, kernel, **kwargs)
        self.features_num = num
        self.resolution_max = resolution_max
        self.resolution_min = resolution_min
        self.dim_widths = domain.statespace_width
        self.centers = np.zeros((num, len(self.dim_widths)))
        self.widths = np.zeros((num, len(self.dim_widths)))
        self.set_seed(seed)

    def set_seed(self, seed):
        super().set_seed(seed)
        for i in range(self.features_num):
            for d in range(len(self.dim_widths)):
                self.centers[i, d] = self.random_state.uniform(
                    self.domain.statespace_limits[d, 0],
                    self.domain.statespace_limits[d, 1],
                )
                self.widths[i, d] = self.random_state.uniform(
                    self.dim_widths[d] / self.resolution_max,
                    self.dim_widths[d] / self.resolution_min,
                )
