"""OMP-TD implementation based on ICML 2012 paper of Wakefield and Parr."""
from copy import deepcopy
from itertools import product
import numpy as np
from rlpy.tools import plt
from .ifdd import iFDD
from .representation import Representation

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class OMPTD(Representation):
    """OMP-TD implementation based on ICML 2012 paper of Wakefield and Parr.
    This implementation assumes an initial representation exists and the bag
    of features is the conjunctions of existing features.
    OMP-TD uses iFDD to represents its features, yet its discovery method is
    different; while iFDD looks at the fringe of the tree of expanded features,
    OMPTD only looks through a predefined set of features.

    The set of features used by OMPTD aside from the initial_features are
    represented by self.expandedFeatures

    """

    IS_DYNAMIC = True
    SHOW_RELEVANCES = False  # Plot the relevances

    def __init__(
        self,
        domain,
        initial_representation,
        discretization=20,
        max_batch_discovery=1,
        batch_threshold=0,
        bag_size=100000,
        sparsify=False,
    ):
        """
        :param domain: the :py:class`~rlpy.domains.domain.Domain` associated
            with the value function we want to learn.
        :param initial_representation: The initial set of features available.
            OMP-TD does not dynamically introduce any features of its own,
            instead it takes conjunctions of initial_representation feats until
            all permutations have been created or bag_size has been reached.
            OMP-TD uses an (ever-growing) subset,termed the \"active\" features.
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        :param max_batch_discovery: Maximum number of features to be expanded on
            each iteration
        :param batch_threshold: Minimum features \"relevance\" required to add
            a feature to the active set.
        :param bag_size: The maximum number of features available for
            consideration.
        :param sparsify: (Boolean)
            See :py:class`~rlpy.representations.ifdd.iFDD`.

        """
        super().__init__(domain, initial_representation.features_num, discretization)
        # This is dummy since omptd will not use ifdd in the online fashion
        self.iFDD_ONLINETHRESHOLD = 1
        self.max_batch_discovery = max_batch_discovery
        self.batch_threshold = batch_threshold
        self.initial_representation = initial_representation
        self.iFDD = iFDD(
            domain,
            self.iFDD_ONLINETHRESHOLD,
            initial_representation,
            sparsify=0,
            discretization=discretization,
            useCache=1,
        )
        self.bag_size = self.fill_bag(bag_size)
        self.total_feature_size = self.bag_size
        # List of selected features. In this implementation initial features are
        # selected initially by default
        self.selected_features = list(range(self.features_num))
        # Array of indicies of features that have not been selected
        self.remaining_features = np.arange(self.features_num, self.bag_size)

    def phi_non_terminal(self, s):
        F_s = self.iFDD.phi_non_terminal(s)
        return F_s[self.selected_features]

    def show(self):
        self.logger.info("Features:\t\t%d" % self.features_num)
        self.logger.info("Remaining Bag Size:\t%d" % len(self.remaining_features))

    def show_bag(self):
        """
        Displays the non-active features that OMP-TD can select from to add
        to its representation.
        """
        print("Remaining Items in the feature bag:")
        for f in self.remaining_features:
            print("%d: %s" % (f, str(sorted(list(self.iFDD.getFeature(f).f_set)))))

    def calculate_full_phi_normalized(self, states):
        """
        In general for OMPTD it is faster to cache the normalized feature matrix
        at once.  Note this is only valid if possible states do not change over
        execution.  (In the feature matrix, each column is a feature function,
        each row is a state; thus the matrix has rows phi(s1)', phi(s2)', ...).
        """
        p = len(states)
        self.fullphi = np.empty((p, self.total_feature_size))
        o_s = self.domain.state
        for i, s in enumerate(states):
            self.domain.state = s
            if not self.domain.is_terminal(s):
                self.fullphi[i, :] = self.iFDD.phi_non_terminal(s)
        self.domain.state = o_s
        # Normalize features
        for f in range(self.total_feature_size):
            phi_f = self.fullphi[:, f]
            norm_phi_f = np.linalg.norm(phi_f)  # L2-Norm of phi_f
            if norm_phi_f == 0:
                norm_phi_f = 1  # This helps to avoid divide by zero
            self.fullphi[:, f] = phi_f / norm_phi_f

    def batch_discover(self, td_errors, phi, states):
        """
        :param td_errors: p-by-1 vector, error associated with each state
        :param phi: p-by-n matrix, vector-valued feature function evaluated at
            each state.
        :param states: p-by-(statedimension) matrix, each state under test.

        Discovers features using OMPTD
        1. Find the index of remaining features in the bag \n
        2. Calculate the inner product of each feature with the TD_Error vector \n
        3. Add the top max_batch_discovery features to the selected features \n

        OUTPUT: Boolean indicating expansion of features
        """
        if len(self.remaining_features) == 0:
            # No More features to Expand
            return False

        self.calculate_full_phi_normalized(states)

        relevances = np.zeros(len(self.remaining_features))
        for i, f in enumerate(self.remaining_features):
            phi_f = self.fullphi[:, f]
            relevances[i] = np.abs(np.dot(phi_f, td_errors))

        if self.SHOW_RELEVANCES:
            e_vec = relevances.flatten()
            e_vec = e_vec[e_vec != 0]
            e_vec = np.sort(e_vec)
            plt.plot(e_vec, linewidth=3)
            plt.ioff()
            plt.show()
            plt.ion()

        # Sort based on relevances
        # We want high to low hence the reverse: [::-1]
        sortedIndices = np.argsort(relevances)[::-1]
        max_relevance = relevances[sortedIndices[0]]

        # Add top <maxDiscovery> features
        self.logger.debug("OMPTD Batch: Max Relevance = %0.3f" % max_relevance)
        added_feature = False
        to_be_deleted = []  # Record the indices of items to be removed
        for j in range(min(self.max_batch_discovery, len(relevances))):
            max_index = sortedIndices[j]
            f = self.remaining_features[max_index]
            relevance = relevances[max_index]
            # print "Inspecting %s" % str(list(self.iFDD.getFeature(f).f_set))
            if relevance >= self.batch_threshold:
                self.logger.debug(
                    "New Feature %d: %s, Relevance = %0.3f"
                    % (
                        self.features_num,
                        str(np.sort(list(self.iFDD.getFeature(f).f_set))),
                        relevances[max_index],
                    )
                )
                to_be_deleted.append(max_index)
                self.selected_features.append(f)
                self.features_num += 1
                added_feature = True
            else:
                # Because the list is sorted, there is no use to look at the
                # others
                break
        self.remaining_features = np.delete(self.remaining_features, to_be_deleted)
        return added_feature

    def fill_bag(self, bag_size):
        """
        Generates potential features by taking conjunctions of existing ones.
        Adds these to the bag of features available to OMPTD in a breadth-first
        fashion until the ``bag_size`` limit is reached.
        """
        level_1_features = np.arange(self.initial_representation.features_num)
        # We store the dimension corresponding to each feature so we avoid
        # adding pairs of features in the same dimension
        level_1_features_dim = []
        for i in range(self.initial_representation.features_num):
            level_1_features_dim.append(
                np.array([self.initial_representation.get_dim_number(i)])
            )
        level_n_features = np.array(level_1_features)
        level_n_features_dim = deepcopy(level_1_features_dim)
        new_id = self.initial_representation.features_num
        self.logger.debug(
            "Added %d size 1 features to the feature bag."
            % (self.initial_representation.features_num)
        )

        # Loop over possible layers that conjunctions can be add. Notice that
        # layer one was already built
        for f_size in np.arange(2, self.domain.state_space_dims + 1):
            added = 0
            next_features = []
            next_features_dim = {}
            for f, g in product(level_1_features, level_n_features):
                f_dim = level_1_features_dim[f][0]
                g_dims = level_n_features_dim[g]
                if f_dim in g_dims:
                    continue
                # We pass inf to make sure iFDD will add the
                # combination of these two features
                if self.iFDD.inspectPair(f, g, np.inf):
                    next_features.append(new_id)
                    next_features_dim[new_id] = g_dims + f_dim
                    new_id += 1
                    added += 1
                    if new_id == bag_size:
                        self.logger.debug(
                            "Added {} size {} features to the feature bag.".format(
                                added, f_size
                            )
                        )
                        break
            level_n_features = next_features
            level_n_features_dim = next_features_dim
            self.logger.debug(
                "Added %d size %d features to the feature bag." % (added, f_size)
            )
        return new_id

    def feature_type(self):
        return self.initial_representation.feature_type()
