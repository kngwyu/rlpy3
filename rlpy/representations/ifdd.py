"""
Incremental Feature Dependency Discovery
iFDD implementation based on ICML 2011 paper
"""
from copy import deepcopy
import numpy as np

from rlpy.tools import printClass, PriorityQueueWithNovelty
from rlpy.tools import powerset, combinations, add_new_features
from rlpy.tools import plt
from .representation import Representation
import warnings
from collections import defaultdict


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


class iFDD_feature:
    """
    This object represents a feature used for linear function approximation.
    The feature can be an initial feature or made using
    the conjunction of existing features.
    :params potential: iFDD_Potential to be used, or a single integer passed
        to generate initial feature.

    TODO: refactor these docs.
    members:
        index: is the location of the feature in the feature vector
        f_set: set of features that their conjunction creates this feature.
        p1, p2: parents of the current feature.
                Notice that multiple combinations of parents may lead to
                the same feature. For example given 3 basic features of X, Y, and Z.
                P1=XY and P2=XZ and P1=X and P2=XZ both lead to the new features XYZ.
                Both these parameters are used for visualizations only.
    """

    def __init__(self, potential):
        if isinstance(potential, iFDD_potential):
            self.index = potential.index
            self.f_set = deepcopy(potential.f_set)
            self.p1 = potential.p1
            self.p2 = potential.p2
        else:
            # This is the case where a single integer is passed to generate an
            # initial feature
            index = potential
            self.index = index
            self.f_set = frozenset([index])
            self.p1 = -1
            self.p2 = -1

    def __deepcopy__(self, memo):
        new_f = iFDD_feature(self.index)
        new_f.p1 = self.p1
        new_f.p2 = self.p2
        new_f.f_set = deepcopy(self.f_set)
        return new_f

    def show(self):
        printClass(self)


class iFDD_potential(object):
    """
    This object represents a potential feature that can be promoted
    to a permanent feature.
    The structure of this object is very similar to iFDD feature object,
    except it has a relevance parameter that measures the importance
    of creating a new feature out of this potential feature.
    :param f_set: Set of features it corresponds to [Immutable].
    :param parant1: Parent 1 feature index.
    :param parant2: Parent 2 feature index.
    """

    def __init__(self, f_set, parent1, parent2):
        self.f_set = deepcopy(f_set)
        self.p1 = parent1
        self.p2 = parent2
        # If this feature has been discovered set this to its index else -1
        self.index = -1
        self.cumtderr = 0
        self.cumabstderr = 0
        self.count = 1  # Number of times this feature was visited

    def __deepcopy__(self, memo):
        new_p = iFDD_potential(self.f_set, self.p1, self.p2)
        new_p.cumtderr = self.cumtderr
        new_p.cumabstderr = self.cumabstderr
        return new_p

    def show(self):
        printClass(self)


class iFDD(Representation):
    """ The incremental Feature Dependency Discovery Representation based on
    [Geramifard et al. 2011 ICML paper]. This representation starts with a set of given
    binary features and adds new features as the conjunction of existing features.
    Given n features iFDD can expand the set of features up to 2^n-1 features
    (i.e. conjunction of each subset of n features can be considered as a new feature.
    """

    IS_DYNAMIC = True

    def __init__(
        self,
        domain,
        discovery_threshold,
        initial_representation,
        sparsify=True,
        discretization=20,
        debug=0,
        useCache=False,
        max_batch_discovery=1,
        batch_threshold=0,
        iFDDPlus=True,
        seed=1,
        use_chirstoph_ordered_features=True,
    ):
        """
        :param domain: The problem :py:class:`~rlpy.domains.domain.Domain` to learn.
        :param discovery_threshold: Psi in the paper.
        :param initial_representation: A Representation that provides the initial
            set of features for iFDD.
        :param sparsify: A boolean specifying the use of the trick mentioned in the
            paper so that features are getting sparser with more feature discoveries
            (i.e. use greedy algorithm for feature activation).
        :param discretization: Number of bins used for each continuous dimension.
        :param debug: Print more stuff.
        :param useCache: Use cache for phi(s) or not.
        :param batch_threshold: Minimum value of feature relevance for
            the batch setting.
        :param max_batch_discovery: Number of features to be expanded in the batch
            setting.
        :param iFDDPlus: ICML 11 iFDD would add sum of abs(TD-errors) while the iFDD
            plus uses the abs(sum(TD-Error))/sqrt(potential feature presence count).
        :param use_chirstoph_ordered_features: As Christoph mentioned adding new
            features may affect the phi for all states.
            This idea was to make sure both conditions for generating active
            features generate the same result.
        """
        super().__init__(
            domain, initial_representation.features_num, discretization, seed
        )
        # dictionary mapping initial feature sets to iFDD_feature
        self.iFDD_features = {}
        # dictionary mapping initial feature sets to iFDD_potential
        self.iFDD_potentials = {}
        # dictionary mapping each feature index (ID) to its feature object
        self.featureIndex2feature = {}
        # dictionary mapping  initial active feature set phi_0(s) to its corresponding
        # active features at phi(s). Based on Tuna's Trick to speed up iFDD
        self.cache = {}
        self.discovery_threshold = discovery_threshold
        self.sparsify = sparsify
        self.set_bins_per_dim(domain, discretization)
        self.debug = debug
        self.useCache = useCache
        self.max_batch_discovery = max_batch_discovery
        self.batch_threshold = batch_threshold
        # This is a priority queue based on the size of the features (Largest ->
        # Smallest). For same size features, it is also sorted based on the newest
        # -> oldest. Each element is the pointer to feature object.
        self.sortediFDDFeatures = PriorityQueueWithNovelty()
        self.initial_representation = initial_representation
        self.iFDDPlus = iFDDPlus
        self.addInitialFeatures()
        self.use_chirstoph_ordered_features = use_chirstoph_ordered_features
        # Helper parameter to get a sense of appropriate threshold on the
        # relevance for discovery
        self.maxRelevance = -np.inf

    def phi_non_terminal(self, s):
        """ Based on Tuna's Master Thesis 2012 """
        F_s = np.zeros(self.features_num, "bool")
        F_s_0 = self.initial_representation.phi_non_terminal(s)
        activeIndices = np.where(F_s_0 != 0)[0]
        if self.useCache:
            finalActiveIndices = self.cache.get(frozenset(activeIndices))
            if finalActiveIndices is None:
                # run regular and update the cache
                finalActiveIndices = self.findFinalActiveFeatures(activeIndices)
        else:
            finalActiveIndices = self.findFinalActiveFeatures(activeIndices)
        F_s[finalActiveIndices] = 1
        return F_s

    def findFinalActiveFeatures(self, intialActiveFeatures):
        """
        Given the active indices of phi_0(s) find the final active indices of phi(s)
        based on discovered features
        """
        finalActiveFeatures = []
        k = len(intialActiveFeatures)
        initialSet = set(intialActiveFeatures)

        if 2 ** k <= self.features_num:
            # k can be big which can cause this part to be very slow
            # if k is large then find active features by enumerating on the
            # discovered features.
            if self.use_chirstoph_ordered_features:
                for i in range(k, 0, -1):
                    if len(initialSet) == 0:
                        break
                    # generate list of all combinations with i elements
                    cand_i = [
                        (c, self.iFDD_features[frozenset(c)].index)
                        for c in combinations(initialSet, i)
                        if frozenset(c) in self.iFDD_features
                    ]
                    # sort (recent features (big ids) first)
                    cand_i.sort(key=lambda x: x[1], reverse=True)
                    for candidate, ind in cand_i:
                        # No more initial features to be mapped to extended ones
                        if len(initialSet) == 0:
                            break

                        # This was missing from ICML 2011 paper algorithm.
                        # Example: [0,1,20], [0,20] is discovered, but if [0]
                        # is checked before [1] it will be added even though it
                        # is already covered by [0,20]
                        if initialSet.issuperset(set(candidate)):
                            feature = self.iFDD_features.get(frozenset(candidate))
                            if feature is not None:
                                finalActiveFeatures.append(feature.index)
                                if self.sparsify:
                                    # print "Sets:", initialSet, feature.f_set
                                    initialSet = initialSet - feature.f_set
                                    # print "Remaining Set:", initialSet
            else:
                for candidate in powerset(initialSet, ascending=0):
                    # No more initial features to be mapped to extended ones
                    if len(initialSet) == 0:
                        break

                    # This was missing from ICML 2011 paper algorithm. Example:
                    # [0,1,20], [0,20] is discovered, but if [0] is checked
                    # before [1] it will be added even though it is already
                    # covered by [0,20]
                    if initialSet.issuperset(set(candidate)):
                        feature = self.iFDD_features.get(frozenset(candidate))
                        if feature is not None:
                            finalActiveFeatures.append(feature.index)
                            if self.sparsify:
                                initialSet = initialSet - feature.f_set
        else:
            # Loop on all features sorted on their size and then novelty and
            # activate features
            for feature in self.sortediFDDFeatures.toList():
                # No more initial features to be mapped to extended ones
                if len(initialSet) == 0:
                    break

                if initialSet.issuperset(set(feature.f_set)):
                    finalActiveFeatures.append(feature.index)
                    if self.sparsify:
                        initialSet = initialSet - feature.f_set

        if self.useCache:
            self.cache[frozenset(intialActiveFeatures)] = finalActiveFeatures
        return finalActiveFeatures

    def post_discover(self, s, terminal, a, td_error, phi_s):
        """
        returns the number of added features
        """
        # Indices of non-zero elements of vector phi_s
        activeFeatures = phi_s.nonzero()[0]
        discovered = 0
        for g_index, h_index in combinations(activeFeatures, 2):
            discovered += self.inspectPair(g_index, h_index, td_error)
        return discovered

    def inspectPair(self, g_index, h_index, td_error):
        """
        Inspect feature f = g union h where g_index and h_index are the indices of
        features g and h.
        If the relevance is > Threshold, add it to the list of features.
        Returns True if a new feature is added.
        """
        g = self.featureIndex2feature[g_index].f_set
        h = self.featureIndex2feature[h_index].f_set
        f = g.union(h)
        feature = self.iFDD_features.get(f)
        if not self.iFDDPlus:
            td_error = abs(td_error)

        if feature is not None:
            # Already exists
            return False

        # Look it up in potentials
        potential = self.iFDD_potentials.get(f)
        if potential is None:
            # Generate a new potential and put it in the dictionary
            potential = iFDD_potential(f, g_index, h_index)
            self.iFDD_potentials[f] = potential

        potential.cumtderr += td_error
        potential.cumabstderr += abs(td_error)
        potential.count += 1
        # Check for discovery

        if self.random_state.rand() < self.iFDDPlus:
            relevance = abs(potential.cumtderr) / np.sqrt(potential.count)
        else:
            relevance = potential.cumabstderr

        if relevance >= self.discovery_threshold:
            self.maxRelevance = -np.inf
            self.addFeature(potential)
            return True
        else:
            self.updateMaxRelevance(relevance)
            return False

    def show(self):
        self.showFeatures()
        self.showPotentials()
        self.showCache()

    def updateWeight(self, p1_index, p2_index):
        """
        Add a new weight corresponding to the new added feature for all actions.
        The new weight is set to zero if sparsify = False, and equal to the
        sum of weights corresponding to the parents if sparsify = True
        """
        if self.sparsify:
            new_elem = self.weight[:, p1_index] + self.weight[:, p2_index]
        else:
            new_elem = None
        self.weight = add_new_features(self.weight, new_elem)
        # We dont want to reuse the hased phi because phi function is changed!
        self.hashed_s = None

    def addInitialFeatures(self):
        for i in range(self.initial_representation.features_num):
            feature = iFDD_feature(i)
            # shout(self,self.iFDD_features[frozenset([i])].index)
            self.iFDD_features[frozenset([i])] = feature
            self.featureIndex2feature[feature.index] = feature
            # priority is 1/number of initial features corresponding to the
            # feature
            priority = 1
            self.sortediFDDFeatures.push(priority, feature)

    def addFeature(self, potential):
        # Add it to the list of features
        # Features_num is always one more than the max index (0-based)
        potential.index = self.features_num
        self.features_num += 1
        feature = iFDD_feature(potential)
        self.iFDD_features[potential.f_set] = feature
        # Expand the size of the weight_vec
        self.updateWeight(feature.p1, feature.p2)
        # Update the index to feature dictionary
        self.featureIndex2feature[feature.index] = feature
        # Update the sorted list of features
        # priority is 1/number of initial features corresponding to the feature
        priority = 1 / len(potential.f_set)
        self.sortediFDDFeatures.push(priority, feature)

        # If you use cache, you should invalidate entries that their initial
        # set contains the set corresponding to the new feature
        if self.useCache:
            for initialActiveFeatures in list(self.cache.keys()):
                if initialActiveFeatures.issuperset(feature.f_set):
                    if self.sparsify:
                        self.cache.pop(initialActiveFeatures)
                    else:
                        # If sparsification is not used, simply add the new
                        # feature id to all cached values that have feature set
                        # which is a super set of the features corresponding to
                        # the new discovered feature
                        self.cache[initialActiveFeatures].append(feature.index)
        if self.debug:
            self.show()

    def batch_discover(self, td_errors, phi, states):
        """
        Discovers features using iFDD in batch setting.
        self.batch_threshold is the minimum relevance value for the feature
        to be expanded.
        :param td_errors: p-by-1 (How much error observed for each sample)
        :param phi: n-by-p features corresponding to all samples
            (each column corresponds to one sample).
        """
        maxDiscovery = self.max_batch_discovery
        n = self.features_num  # number of features
        p = len(td_errors)  # Number of samples
        counts = np.zeros((n, n))
        relevances = np.zeros((n, n))
        for i in range(p):
            phiphiT = np.outer(phi[i, :], phi[i, :])
            if self.iFDDPlus:
                relevances += phiphiT * td_errors[i]
            else:
                relevances += phiphiT * abs(td_errors[i])
            counts += phiphiT
        # Remove Diagonal and upper part of the relevances as they are useless
        relevances = np.triu(relevances, 1)
        non_zero_index = np.nonzero(relevances)
        if self.iFDDPlus:
            # Calculate relevances based on theoretical results of ICML 2013
            # potential submission
            relevances[non_zero_index] = np.divide(
                np.abs(relevances[non_zero_index]), np.sqrt(counts[non_zero_index])
            )
        else:
            # Based on Geramifard11_ICML Paper
            relevances[non_zero_index] = relevances[non_zero_index]

        # Find indexes to non-zero excited pairs
        # F1 and F2 are the parents of the potentials
        (F1, F2) = relevances.nonzero()
        relevances = relevances[F1, F2]
        if len(relevances) == 0:
            # No feature to add
            self.logger.debug("iFDD Batch: Max Relevance = 0")
            return False

        if self.debug:
            e_vec = relevances.flatten()
            e_vec = e_vec[e_vec != 0]
            e_vec = np.sort(e_vec)
            plt.ioff()
            plt.plot(e_vec, linewidth=3)
            plt.show()

        # Sort based on relevances
        # We want high to low hence the reverse: [::-1]
        sortedIndices = np.argsort(relevances)[::-1]
        max_relevance = relevances[sortedIndices[0]]
        # Add top <maxDiscovery> features
        self.logger.debug("iFDD Batch: Max Relevance = {0:g}".format(max_relevance))
        added_feature = False
        new_features = 0
        for j in range(len(relevances)):
            if new_features >= maxDiscovery:
                break
            max_index = sortedIndices[j]
            f1 = F1[max_index]
            f2 = F2[max_index]
            relevance = relevances[max_index]
            if relevance > self.batch_threshold:
                # print "Inspecting",
                # f1,f2,'=>',self.getStrFeatureSet(f1),self.getStrFeatureSet(f2)
                if self.inspectPair(f1, f2, np.inf):
                    self.logger.debug(
                        "New Feature %d: %s, Relevance = %0.3f"
                        % (
                            self.features_num - 1,
                            self.getStrFeatureSet(self.features_num - 1),
                            relevances[max_index],
                        )
                    )
                    new_features += 1
                    added_feature = True
            else:
                # Because the list is sorted, there is no use to look at the
                # others
                break
        # A signal to see if the representation has been expanded or not
        return added_feature

    def showFeatures(self):
        print("Features:")
        print("-" * 30)
        print(" index\t| f_set\t| p1\t| p2\t | Weights (per action)")
        print("-" * 30)
        for feature in reversed(self.sortediFDDFeatures.toList()):
            print(
                " %d\t| %s\t| %s\t| %s\t| Omitted"
                % (
                    feature.index,
                    self.getStrFeatureSet(feature.index),
                    feature.p1,
                    feature.p2,
                )
            )

    def showPotentials(self):
        print("Potentials:")
        print("-" * 30)
        print(" index\t| f_set\t| relevance\t| count\t| p1\t| p2")
        print("-" * 30)
        for _, potential in self.iFDD_potentials.items():
            print(
                " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s"
                % (
                    potential.index,
                    str(np.sort(list(potential.f_set))),
                    potential.relevance,
                    potential.count,
                    potential.p1,
                    potential.p2,
                )
            )

    def showCache(self):
        if self.useCache:
            print("Cache:")
            if len(self.cache) == 0:
                print("EMPTY!")
                return
            print("-" * 30)
            print(" initial\t| Final")
            print("-" * 30)
            for initial, active in self.cache.items():
                print(" %s\t| %s" % (str(list(initial)), active))

    def updateMaxRelevance(self, newRelevance):
        # Update a global max relevance and outputs it if it is updated
        if self.maxRelevance < newRelevance:
            self.maxRelevance = newRelevance
            if self.debug:
                self.logger.debug(
                    "iFDD Batch: Max Relevance = {0:g}".format(newRelevance)
                )

    def getFeature(self, f_id):
        # returns a feature given a feature id
        if f_id in list(self.featureIndex2feature.keys()):
            return self.featureIndex2feature[f_id]
        else:
            print("F_id %d is not valid" % f_id)
            return None

    def getStrFeatureSet(self, f_id):
        # returns a string that corresponds to the set of features specified by
        # the given feature_id
        if f_id in list(self.featureIndex2feature.keys()):
            return str(sorted(list(self.featureIndex2feature[f_id].f_set)))
        else:
            print("F_id %d is not valid" % f_id)
            return None

    def feature_type(self):
        return bool

    def __deepcopy__(self, memo):
        ifdd = iFDD(
            self.domain,
            self.discovery_threshold,
            self.initial_representation,
            self.sparsify,
            self.discretization,
            self.debug,
            self.useCache,
            self.max_batch_discovery,
            self.batch_threshold,
            self.iFDDPlus,
        )
        for s, f in list(self.iFDD_features.items()):
            new_f = deepcopy(f)
            new_s = deepcopy(s)
            ifdd.iFDD_features[new_s] = new_f
            ifdd.featureIndex2feature[new_f.index] = new_f
        for s, p in list(self.iFDD_potentials.items()):
            new_s = deepcopy(s)
            new_p = deepcopy(p)
            ifdd.iFDD_potentials[new_s] = deepcopy(new_p)
        ifdd.cache = deepcopy(self.cache)
        ifdd.sortediFDDFeatures = deepcopy(self.sortediFDDFeatures)
        ifdd.features_num = self.features_num
        ifdd.weight = deepcopy(self.weight)
        return ifdd


class iFDDK_potential(iFDD_potential):
    def __init__(self, f_set, parent1, parent2):
        """
        :param f_set: Set of features it corresponds to [Immutable].
        :param parant1: Parent 1 feature index.
        :param parant2: Parent 2 feature index.
        """
        self.f_set = deepcopy(f_set)
        # If this feature has been discovered set this to its index else -1
        self.index = -1
        self.p1 = parent1
        self.p2 = parent2
        try:
            self.hp_dtype = np.dtype("float128")
        except TypeError:
            self.hp_dtype = np.dtype("float64")
        self.a = np.array(0.0, dtype=self.hp_dtype)  # tE[phi |\delta|] estimate
        self.b = np.array(0.0, dtype=self.hp_dtype)  # tE[phi \delta] estimate
        self.e = np.array(0.0, dtype=self.hp_dtype)  # eligibility trace
        self.n_crho = 0  # rho episode index of last update
        self.last_stats = {}

    def relevance(self, kappa=None, plus=None):
        if plus is None:
            assert kappa is not None
            plus = self.random_state.rand() >= kappa

        if plus:
            return np.abs(self.b) / np.sqrt(self.c)
        else:
            return self.a / np.sqrt(self.c)

    def update_statistics(self, rho, td_error, lambda_, discount_factor, phi, n_rho):
        # phi = phi_s[self.p1] * phi_s[self.p2]
        if n_rho > self.n_crho:
            self.e = 0
        self.e = rho * (lambda_ * discount_factor * self.e + phi)

        self.a += np.abs(td_error) * self.e
        self.b += td_error * self.e
        self.c += phi ** 2

        self.n_crho = n_rho

    def update_lazy_statistics(
        self, rho, td_error, lambda_, discount_factor, phi, y_a, y_b, t_rho, w, t, n_rho
    ):
        # phi = phi_s[self.p1] * phi_s[self.p2]
        if lambda_ > 0:
            # catch up on old updates
            gl = np.power(
                np.array(discount_factor * lambda_, dtype=self.hp_dtype),
                t_rho[n_rho] - self.l,
            )
            sa, sb = self.a.copy(), self.b.copy()
            self.a += self.e * (y_a[self.n_crho] - self.x_a) * np.exp(-self.nu) * gl
            self.b += self.e * (y_b[self.n_crho] - self.x_b) * np.exp(-self.nu) * gl
            if not np.isfinite(self.a):
                self.a = sa
                warnings.warn("Overflow in potential relevance estimate")
            if not np.isfinite(self.b):
                self.b = sb
                warnings.warn("Overflow in potential relevance estimate")
            if n_rho > self.n_crho:
                self.e = 0
                # TODO clean up y_a and y_b
            else:
                self.e *= (
                    gl
                    * (lambda_ * discount_factor) ** (t - 1 - t_rho[n_rho])
                    * np.exp(w - self.nu)
                )

        # updates based on current transition
        # np.seterr(under="warn")
        self.e = rho * (lambda_ * discount_factor * self.e + phi)

        self.a += np.abs(td_error) * self.e
        self.b += td_error * self.e
        # np.seterr(under="raise")
        self.c += phi ** 2
        # save current values for next catch-up update
        self.last_stats["t"] = t
        self.last_stats["y_a"] = y_a[n_rho].copy()
        self.last_stats["y_b"] = y_b[n_rho].copy()
        self.last_stats["w"] = w
        self.n_crho = n_rho

    def __deepcopy__(self, memo):
        new_p = iFDD_potential(self.f_set, self.p1, self.p2)
        for i in ["a", "b", "c", "n_crho", "e", "last_stats"]:
            new_p.__dict__[i] = deepcopy(self.__dict__[i])
        return new_p


def _set_comp_lt(a, b):
    if len(a) < len(b):
        return -1
    elif len(a) > len(b):
        return 1
    l1 = list(a)
    l2 = list(b)
    l1.sort()
    l2.sort()
    for i in range(len(l1)):
        if l1[i] < l2[i]:
            return -1
        if l1[i] > l2[i]:
            return +1
    return 0


class iFDDK(iFDD):
    """iFDD(kappa) algorithm with support for elibility traces
    The iFDD(kappa) algorithm is a stochastic mixture of iFDD and iFDD+
    to retain the best properties of both algorithms.
    """

    def __init__(
        self,
        domain,
        discovery_threshold,
        initial_representation,
        sparsify=True,
        discretization=20,
        debug=0,
        useCache=0,
        kappa=1e-5,
        lambda_=0.0,
        lazy=False,
    ):
        try:
            self.hp_dtype = np.dtype("float128")
        except TypeError:
            self.hp_dtype = np.dtype("float64")
        self.y_a = defaultdict(lambda: np.array(0.0, dtype=self.hp_dtype))
        self.y_b = defaultdict(lambda: np.array(0.0, dtype=self.hp_dtype))
        self.lambda_ = lambda_
        self.kappa = kappa
        self.discount_factor = domain.discount_factor
        self.lazy = lazy  # lazy updating?
        self.w = 0  # log(rho) trace
        self.n_rho = 0  # index for rho episodes
        self.t = 0
        self.t_rho = defaultdict(int)
        super().__init__(
            domain,
            discovery_threshold,
            initial_representation,
            sparsify=sparsify,
            discretization=discretization,
            debug=debug,
            useCache=useCache,
        )

    def episode_terminated(self):
        self.n_rho += 1
        self.w = 0
        self.t_rho[self.n_rho] = self.t

    def showPotentials(self):
        print("Potentials:")
        print("-" * 30)
        print(" index\t| f_set\t| relevance\t| count\t| p1\t| p2")
        print("-" * 30)

        k = sorted(iter(self.iFDD_potentials.keys()), cmp=_set_comp_lt)
        for f_set in k:
            potential = self.iFDD_potentials[f_set]
            print(
                " %d\t| %s\t| %g\t| %d\t| %s\t| %s"
                % (
                    potential.index,
                    str(np.sort(list(potential.f_set))),
                    potential.relevance(plus=True),
                    potential.c,
                    potential.p1,
                    potential.p2,
                )
            )

    def post_discover(self, s, terminal, a, td_error, phi_s, rho=1):
        """
        returns the number of added features
        """
        self.t += 1
        discovered = 0
        plus = self.random_state.rand() >= self.kappa
        # if self.t == 22:
        #    import ipdb; ipdb.set_trace()
        activeFeatures = phi_s.nonzero()[
            0
        ]  # Indices of non-zero elements of vector phi_s
        # print "Active Features:", activeFeatures, phi_s
        if not self.lazy:
            dd = defaultdict(float)
            for g_index, h_index in combinations(activeFeatures, 2):
                # create potential if necessary
                dd[self.get_potential(g_index, h_index).f_set] = (
                    phi_s[g_index] * phi_s[h_index]
                )

            for f, potential in list(self.iFDD_potentials.items()):
                potential.update_statistics(
                    rho, td_error, self.lambda_, self.discount_factor, dd[f], self.n_rho
                )
                # print plus, potential.relevance(plus=plus)
                if potential.relevance(plus=plus) >= self.discovery_threshold:
                    self.addFeature(potential)
                    del self.iFDD_potentials[f]
                    discovered += 1

            return discovered

        for g_index, h_index in combinations(activeFeatures, 2):
            discovered += self.inspectPair(g_index, h_index, td_error, phi_s, rho, plus)

        if rho > 0:
            self.w += np.log(rho)
        else:
            # cut e-traces
            self.n_rho += 1
            self.w = 0
            self.t_rho[self.n_rho] = self.t

        if self.lambda_ > 0:
            self.y_a[self.n_rho] += (
                np.exp(self.w)
                * (self.discount_factor * self.lambda_)
                ** (self.t - self.t_rho[self.n_rho])
                * np.abs(td_error)
            )
            self.y_b[self.n_rho] += (
                np.exp(self.w)
                * (self.discount_factor * self.lambda_)
                ** (self.t - self.t_rho[self.n_rho])
                * td_error
            )
            assert np.isfinite(self.y_a[self.n_rho])
            assert np.isfinite(self.y_b[self.n_rho])

        return discovered

    def get_potential(self, g_index, h_index):
        g = self.featureIndex2feature[g_index].f_set
        h = self.featureIndex2feature[h_index].f_set
        f = g.union(h)
        feature = self.iFDD_features.get(f)

        if feature is not None:
            # Already exists
            return None

        # Look it up in potentials
        potential = self.iFDD_potentials.get(f)
        if potential is None:
            # Generate a new potential and put it in the dictionary
            potential = iFDDK_potential(f, g_index, h_index)
            self.iFDD_potentials[f] = potential
        return potential

    def inspectPair(self, g_index, h_index, td_error, phi_s, rho, plus):
        """
        Inspect feature f = g union h where g_index and h_index are the indices of
        features g and h.
        If the relevance is > Threshold add it to the list of features.
        Returns True if a new feature is added,
        """
        potential = self.get_potential(g_index, h_index)
        phi = phi_s[g_index] * phi_s[h_index]
        potential.update_lazy_statistics(
            rho,
            td_error,
            self.lambda_,
            self.discount_factor,
            phi,
            self.y_a,
            self.y_b,
            self.t_rho,
            self.w,
            self.t,
            self.n_rho,
        )
        # Check for discovery

        relevance = potential.relevance(plus=plus)
        if relevance >= self.discovery_threshold:
            self.maxRelevance = -np.inf
            self.addFeature(potential)
            del self.iFDD_potentials[potential.f_set]
            return 1
        else:
            self.updateMaxRelevance(relevance)
            return 0
