"""Kernelized Incremental Feature Dependency Discovery"""
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from rlpy.tools import add_new_features, PriorityQueueWithNovelty
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
__author__ = "Christoph Dann <cdann@mit.edu>"


class KernelizedFeature:
    def __init__(self, center, dim, kernel, index=-1, base_ids=None, kernel_args=[]):
        """
        :param center: Data point used to generate the feature, which
            gives the highest output of this feature.
        :param dim: List of dimensions that are regarded by this feature.
        :param kernel: The kernel used to generate featues.
        :param index: The index of the feature. -1 is for non-discovered ones.
        """
        self.index = index
        self.kernel_args = kernel_args
        self.center = center
        self.dim = dim
        self.kernel = kernel
        if base_ids is None:
            self.base_ids = frozenset([self.index])
        else:
            self.base_ids = base_ids

    def __str__(self):
        res = "{" + ", ".join(sorted([str(i) for i in self.base_ids])) + "}  "
        res += ", ".join(["s{}={:.3g}".format(d + 1, self.center[d]) for d in self.dim])
        return res

    def output(self, s):
        return self.kernel(s, self.center, self.dim, *self.kernel_args)

    def key(self):
        return len(self.base_ids), tuple(self.dim), tuple(self.center[self.dim])


class Candidate(object):
    """
    candidate feature as a combination of two existing features
    """

    def __init__(self, idx1, idx2):
        self.idx1 = idx1
        self.idx2 = idx2
        self.activation_count = 0.0
        self.td_error_sum = 0.0
        self.relevance = 0.0


class KernelizediFDD(Representation):
    """
    Kernelized version of iFDD
    """

    def __init__(
        self,
        domain,
        kernel,
        active_threshold,
        discover_threshold,
        kernel_args=[],
        normalization=True,
        sparsify=True,
        max_active_base_feat=2,
        max_base_feat_sim=0.7,
    ):
        super().__init__(domain, 0)
        self.kernel = kernel
        self.kernel_args = kernel_args
        self.active_threshold = active_threshold
        self.discover_threshold = discover_threshold
        self.normalization = normalization
        self.sparsify = sparsify
        self.sorted_ids = PriorityQueueWithNovelty()
        self.max_active_base_feat = max_active_base_feat
        self.max_base_feat_sim = max_base_feat_sim
        self.candidates = {}
        self.features = []
        self.base_features_ids = []
        self.max_relevance = 0.0
        # contains a set for each feature indicating the ids of
        self.base_id_sets = set()
        self.base_feature_ids = []  # 1-dim features it rnefines

    def show_features(self):
        ids = self.sorted_ids.toList()[:]
        ids.sort(key=lambda x: self.features[x].key())
        for i in ids:
            f = self.features[i]
            print("{:>5} {:>20}".format(i, f))

    def plot_1d_features(self, dimension_idx=None):
        """Creates a plot for each specified dimension of the state space and shows
        all 1-dimensional features in this dimension
        If no indices are passed, all dimensions are plotted

        dimension_idx: either a single dimension index (int) or a list of indices.
        """
        idx = dimension_idx
        if isinstance(idx, int):
            idx = [idx]
        elif idx is None:
            idx = self.domain.continuous_dims

        feat_list = list(range(self.features_num))
        feat_list.sort(key=lambda x: self.features[x].key())
        last_i = -1
        for k in feat_list:
            if len(self.features[k].dim) > 1:
                break
            cur_i = self.features[k].dim[0]
            if cur_i != last_i:
                if last_i in idx:
                    plt.draw()
                if cur_i in idx:
                    xi = np.linspace(
                        self.domain.statespace_limits[cur_i, 0],
                        self.domain.statespace_limits[cur_i, 1],
                        200,
                    )
                    x = np.zeros((200, self.domain.statespace_limits.shape[0]))
                    x[:, cur_i] = xi
                    plt.figure("Feature Dimension {}".format(cur_i))
            if cur_i in idx:
                y = [self.features[k].output(xk) for xk in x]
                plt.plot(x, y, label="id {}".format(k))
            last_i = cur_i
        plt.draw()

    def plot_2d_features(self, d1=None, d2=None, n_lines=3):
        """
        plot contours of all 2-dimensional features covering
        dimension d1 and d2. For each feature, n_lines number of lines
        are shown.
        If no dimensions are specified, the first two continuous dimensions
        are shown.

        d1, d2: indices of dimensions to show
        n_lines: number of countour lines per feature (default: 3)
        """
        if d1 is None and d2 is None:
            # just take the first two dimensions
            idx = self.domain.continuous_dims[:2]
        else:
            idx = [d1, d2]
        idx.sort()

        feat_list = list(range(self.features_num))
        feat_list.sort(key=lambda x: self.features[x].key())
        last_i = -1
        last_j = -1
        for k in feat_list:
            if len(self.features[k].dim) < 2:
                continue
            elif len(self.features[k].dim) > 2:
                break
            cur_i = self.features[k].dim[0]
            cur_j = self.features[k].dim[1]
            if cur_i != last_i or cur_j != last_j:
                if last_i in idx and last_j in idx:
                    plt.draw()
                if cur_i in idx and cur_j in idx:
                    xi = np.linspace(
                        self.domain.statespace_limits[cur_i, 0],
                        self.domain.statespace_limits[cur_i, 1],
                        100,
                    )
                    xj = np.linspace(
                        self.domain.statespace_limits[cur_j, 0],
                        self.domain.statespace_limits[cur_j, 1],
                        100,
                    )
                    X, Y = np.meshgrid(xi, xj)
                    plt.figure("Feature Dimensions {} and {}".format(cur_i, cur_j))
            if cur_i in idx and cur_j in idx:
                Z = np.zeros_like(X)
                for m in range(100):
                    for n in range(100):
                        x = np.zeros(self.domain.statespace_limits.shape[0])
                        x[cur_i] = X[m, n]
                        x[cur_j] = Y[m, n]
                        Z[m, n] = self.features[k].output(x)
                plt.contour(X, Y, Z, n_lines)
            last_i = cur_i
            last_j = cur_j
        plt.draw()

    def plot_2d_feature_centers(self, d1=None, d2=None):
        """
        plot the centers of all 2-dimensional features covering
        dimension d1 and d2.
        If no dimensions are specified, the first two continuous dimensions
        are shown.

        d1, d2: indices of dimensions to show
        """
        if d1 is None and d2 is None:
            # just take the first two dimensions
            idx = self.domain.continuous_dims[:2]
        else:
            idx = [d1, d2]
        idx.sort()

        feat_list = list(range(self.features_num))
        feat_list.sort(key=lambda x: self.features[x].key())
        last_i = -1
        last_j = -1
        for k in feat_list:
            if len(self.features[k].dim) < 2:
                continue
            elif len(self.features[k].dim) > 2:
                break
            cur_i = self.features[k].dim[0]
            cur_j = self.features[k].dim[1]
            if cur_i != last_i or cur_j != last_j:
                if last_i in idx and last_j in idx:
                    plt.draw()
                if cur_i in idx and cur_j in idx:
                    plt.figure("Feature Dimensions {} and {}".format(cur_i, cur_j))
            if cur_i in idx and cur_j in idx:
                plt.plot(
                    [self.features[k].center[cur_i]],
                    [self.features[k].center[cur_j]],
                    "r",
                    marker="x",
                )
            last_i = cur_i
            last_j = cur_j
        plt.draw()

    def phi_non_terminal(self, s):
        out = np.zeros(self.features_num)
        if not self.sparsify:
            for i in range(self.features_num):
                out[i] = self.features[i].output(s)
        else:
            # get all base feature values and check if they are activated
            active_bases = set([])
            for i in self.sorted_ids.toList()[::-1]:
                if len(self.features[i].base_ids) > 1:
                    break
                if self.features[i].output(s) >= self.active_threshold:
                    active_bases.add(i)

            base_vals = {k: 1.0 for k in active_bases}
            # iterate over the remaining compound features
            for i in self.sorted_ids.toList():
                if active_bases.issuperset(self.features[i].base_ids):
                    if self.sparsify > 1:
                        out[i] = self.features[i].output(s)
                        if self.sparsify > 2 or out[i] >= self.active_threshold:
                            active_bases -= self.features[i].base_ids
                    else:
                        u = 0
                        for k in self.features[i].base_ids:
                            u = max(u, base_vals[k])
                        out[i] = self.features[i].output(s) * u

                        for k in self.features[i].base_ids:
                            base_vals[k] -= out[i]
                            if base_vals[k] < 0:
                                active_bases.remove(k)

        if self.normalization:
            summ = out.sum()
            if summ != 0:
                out /= out.sum()
        return out

    def phi_raw(self, s, terminal):
        assert terminal is False
        out = np.zeros(self.features_num)
        for i in range(self.features_num):
            out[i] = self.features[i].output(s)
        return out

    # @profile
    def post_discover(self, s, terminal, a, td_error, phi_s=None):
        if phi_s is None:
            phi_s = self.phi(s, terminal)
        phi_s_unnorm = self.phi_raw(s, terminal)
        discovered = 0
        Q = self.Qs(s, terminal, phi_s=phi_s).reshape(-1, 1)
        # indices of active features
        active_indices = list(np.where(phi_s_unnorm > self.active_threshold)[0])
        # "active indices", active_indices
        # gather all dimensions regarded by active features
        active_dimensions = np.zeros((len(s)), dtype="int")
        closest_neighbor = np.zeros((len(s)))
        for i in active_indices:
            for j in self.features[i].dim:
                active_dimensions[j] += 1
                closest_neighbor[j] = max(closest_neighbor[j], phi_s_unnorm[i])

        # add new base features for all dimension not regarded
        for j in range(len(s)):
            if active_dimensions[j] < self.max_active_base_feat and (
                closest_neighbor[j] < self.max_base_feat_sim or active_dimensions[j] < 1
            ):
                active_indices.append(self.add_base_feature(s, j, Q=Q))
                discovered += 1

        # update relevance statistics of all feature candidates
        if discovered:
            phi_s = self.phi(s, terminal)
        la = len(active_indices)
        if la * (la - 1) < len(self.candidates):
            for ind, cand in list(self.candidates.items()):
                g, h = ind
                rel = self.update_relevance_stat(cand, g, h, td_error, s, a, phi_s)
                self.max_relevance = max(rel, self.max_relevance)
                # add if relevance is high enough
                if rel > self.discover_threshold:
                    self.add_refined_feature(g, h, Q=Q)
                    discovered += 1

        else:
            # the result of both branches can be very different as this one
            # updates only combinations which are considered active.
            for g, h in combinations(active_indices, 2):
                # note: g, h are ordered as active_indices are ordered
                cand = self.candidates.get((g, h))
                if cand is None:
                    continue
                rel = self.update_relevance_stat(cand, g, h, td_error, s, a, phi_s)
                self.max_relevance = max(rel, self.max_relevance)
                # add if relevance is high enough
                if rel > self.discover_threshold:
                    self.add_refined_feature(g, h, Q=Q)
                    discovered += 1

        if discovered:
            self.max_relevance = 0.0
        return discovered

    def update_relevance_stat(self, candidate, index1, index2, td_error, s, a, phi_s):
        """
        make sure that inputs are ordered, i.e.,index1 <= index2!
        returns the relevance of a potential feature combination
        """
        candidate.td_error_sum += phi_s[index1] * phi_s[index2] * td_error
        candidate.activation_count += phi_s[index1] ** 2 * phi_s[index2] ** 2
        if candidate.activation_count == 0.0:
            return 0.0
        return np.abs(candidate.td_error_sum) / np.sqrt(candidate.activation_count)

    def add_base_feature(self, center, dim, Q):
        """
        adds a new 1-dimensional feature and returns its index
        """
        new_f = KernelizedFeature(
            center=center,
            dim=[dim],
            kernel_args=self.kernel_args,
            kernel=self.kernel,
            index=self.features_num,
        )
        self.features.append(new_f)

        self.base_id_sets.add(new_f.base_ids)
        self.sorted_ids.push(-1, self.features_num)
        self.logger.debug("Added Feature {} {}".format(self.features_num, new_f))

        # add combinations with all existing features as candidates
        new_cand = {
            (f, self.features_num): Candidate(f, self.features_num)
            for f in range(self.features_num)
            if dim not in self.features[f].dim
        }

        self.candidates.update(new_cand)
        for f, _ in list(new_cand.keys()):
            self.base_id_sets.add(new_f.base_ids | self.features[f].base_ids)
        self.features_num += 1

        # add parameter dimension
        if self.normalization:
            self.weight = add_new_features(self.weight, Q)
        else:
            self.weight = add_new_features(self.weight)
        return self.features_num - 1

    def add_refined_feature(self, index1, index2, Q):
        """
        adds the combination of 2 existing features to the representation
        """
        f1 = self.features[index1]
        f2 = self.features[index2]
        new_center = np.zeros_like(f1.center)
        cnt = np.zeros_like(f1.center)
        cnt[f1.dim] += 1
        cnt[f2.dim] += 1
        cnt[cnt == 0] = 1.0
        new_center[f1.dim] += f1.center[f1.dim]
        new_center[f2.dim] += f2.center[f2.dim]
        new_center /= cnt
        new_dim = list(frozenset(f1.dim) | frozenset(f2.dim))
        new_base_ids = f1.base_ids | f2.base_ids
        new_dim.sort()
        new_f = KernelizedFeature(
            center=new_center,
            dim=new_dim,
            kernel_args=self.kernel_args,
            kernel=self.kernel,
            index=self.features_num,
            base_ids=new_base_ids,
        )
        self.features.append(new_f)
        # Priority is the negative number of base ids
        self.sorted_ids.push(-len(new_f.base_ids), self.features_num)
        # assert(len(self.sorted_ids.toList()) == self.features_num + 1)
        self.base_id_sets.add(new_f.base_ids)
        del self.candidates[(index1, index2)]

        # add new candidates
        new_cand = {
            (f, self.features_num): Candidate(f, self.features_num)
            for f in range(self.features_num)
            if (self.features[f].base_ids | new_base_ids) not in self.base_id_sets
            and len(frozenset(self.features[f].dim) & frozenset(new_dim)) == 0
        }
        for c, _ in list(new_cand.keys()):
            self.base_id_sets.add(new_base_ids | self.features[c].base_ids)
        self.candidates.update(new_cand)
        self.logger.debug(
            "Added refined feature {} {}".format(self.features_num, new_f)
        )
        self.logger.debug("{} candidates".format(len(self.candidates)))
        self.features_num += 1
        if self.normalization:
            self.weight = add_new_features(self.weight, Q)
        else:
            self.weight = add_new_features(self.weight)

        return self.features_num - 1


try:
    from .kernels import *  # noqa
except ImportError:
    import warnings

    warnings.warn("C-Extension for kernels not available, expect slow runtime")
    from .slow_kernels import *  # noqa
