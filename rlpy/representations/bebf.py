"""Bellman-Error Basis Function Representation."""
import numpy as np
from .representation import Representation

try:
    from sklearn import svm
except ImportError:
    import warnings

    warnings.warn("sklearn is not installed and you cannot use BEBF")

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Robert H. Klein"


class BEBF(Representation):

    """Bellman-Error Basis Function Representation.

    .. warning::

        REQUIRES the implementation of locally-weighted
        projection regression (LWPR), available at:
        http://wcms.inf.ed.ac.uk/ipab/slmc/research/software-lwpr

    Parameters set according to: Parr et al.,
    "Analyzing Feature Generation for Function Approximation" (2007).
    http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_ParrPLL07.pdf

    Bellman-Error Basis Function Representation. \n
    1. Initial basis function based on immediate reward. \n
    2. Evaluate r + Q(s', π{s'}) - Q(s,a) for all samples. \n
    3. Train function approximator on bellman error of present solution above\n
    4. Add the above as a new basis function. \n
    5. Repeat the process using the new basis until the most
    recently added basis function has norm <= batch_threshold, which
    Parr et al. used as 10^-5.\n

    Note that the *USER* can select the class of feature functions to be used;
    the BEBF function approximator itself consists of many feature functions
    which themselves are often approximations to their particular functions.
    Default here is to train a support vector machine (SVM) to be used for
    each feature function.
    """

    IS_DYNAMIC = True

    def __init__(
        self,
        domain,
        discretization=20,
        batch_threshold=10 ** -3,
        svm_epsilon=0.1,
        max_batch_discovery=1,
    ):
        """
        :param domain: the problem :py:class:`~rlpy.domains.domain.Domain` to learn
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        :param batch_threshold: Threshold below which no more features are added
            for a given data batch.
        :param svm_epsilon: (From sklearn, scikit-learn): \"epsilon in the
            epsilon-SVR model. It specifies the epsilon-tube within which no
            penalty is associated in the training loss function with points
            predicted within a distance epsilon from the actual value.\"
        :param max_batch_discovery:  Number of features to be expanded in the
            batch setting. By default, it's 1 since each BEBF will be identical
            on a given iteration
        """
        self.set_bins_per_dim(domain, discretization)
        # Effectively initialize with IndependentDiscretization
        self.initial_features_num = int(sum(self.bins_per_dim))
        super().__init__(domain, self.initial_features_num, discretization)
        self.max_batch_discovery = max_batch_discovery
        self.svm_epsilon = svm_epsilon
        self.batch_threshold = batch_threshold
        self.features = []

    def get_function_approximation(self, X, y):
        """
        :param X: Training dataset inputs
        :param y: Outputs associated with training set.

        Accepts dataset (X,y) and trains a feature function on it
        (default uses Support Vector Machine).
        Returns a handle to the trained feature function.
        """
        bebfApprox = svm.SVR(kernel="rbf", degree=3, C=1.0, epsilon=self.svm_epsilon)
        bebfApprox.fit(X, y)
        return bebfApprox

    def phi_non_terminal(self, s):
        F_s = np.zeros(self.features_num)
        # From IndependentDiscretization
        F_s[self.activeInitialFeatures(s)] = 1
        bebf_features_num = self.features_num - self.initial_features_num
        for features_ind, F_s_ind in enumerate(
            np.arange(bebf_features_num) + self.initial_features_num
        ):
            F_s[F_s_ind] = self.features[features_ind].predict(s)
        return F_s

    def batch_discover(self, td_errors, all_phi_s, s):
        """
        Adds new features based on the Bellman Error in batch setting.
        :param td_errors:
        :param td_errors: p-by-1 (How much error observed for each sample)
        :param all_phi_s: n-by-p features corresponding to all samples
            (each column corresponds to one sample).
        :param s: List of states corresponding to each td_error in td_errors
            (note that the same state may appear multiple times
             because of different actions taken while there).
        """
        # need states here instead?
        addedFeature = False
        # PLACEHOLDER for norm of function
        norm = max(abs(td_errors))  # Norm of function
        for j in range(self.max_batch_discovery):
            self.features.append(self.get_function_approximation(s, td_errors))
            if norm > self.batch_threshold:
                self.add_new_weight()
                addedFeature = True
                self.features_num += 1
                self.logger.debug(
                    "Added feature. \t %d total feats" % self.features_num
                )
            else:
                break
        return addedFeature

    def feature_type(self):
        return float
