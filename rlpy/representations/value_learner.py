"""Implementation of value learning utilities
"""
import numpy as np
from rlpy.tools import add_new_features, findElemArray1D


class ValueLearner:
    """
    Have weights for all feature/action pairs and learning algorithms
    """

    def __init__(self, actions_num, features_num):
        try:
            #: A numpy array of the Linear Weights, one for each feature (theta)
            self.weight = np.zeros((actions_num, features_num))
        except MemoryError:
            raise MemoryError(
                "Unable to allocate weights of size: {}\n".format(
                    features_num * actions_num
                )
            )
        self.actions_num = actions_num
        self.features_num = features_num
        self._phi_sa_cache = np.empty((self.actions_num, self.features_num))

    @property
    def weight_vec(self):
        """
        Flat view of weight.
        Exists for backward compatibility.
        """
        return self.weight.reshape(-1)

    @weight_vec.setter
    def weight_vec(self, v):
        self.weight = v.view().reshape(self.weight.shape)

    def add_new_weight(self):
        """
        Add a new zero weight, corresponding to a newly added feature,
        to all actions.
        """
        self.weight = add_new_features(self.weight)

    def V(self, s, terminal, p_actions, phi_s):
        """ Returns the value of state s under possible actions p_actions.

        :param s: The queried state
        :param terminal: Whether or not *s* is a terminal state
        :param p_actions: the set of possible actions
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.

        See :py:meth:`~rlpy.representations.representation.Qs`.
        """
        all_qs = self.Qs(s, terminal, phi_s)
        if len(p_actions) > 0:
            return max(all_qs[p_actions])
        else:
            return 0.0

    def Qs(self, s, terminal, phi_s):
        """
        Returns an array of actions available at a state and their
        associated values.

        :param s: The queried state
        :param terminal: Whether or not *s* is a terminal state
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.

        :return: The tuple (Q,A) where:
            - Q: an array of Q(s,a), the values of each action at *s*. \n
            - A: the corresponding array of actionIDs (integers)
        """
        if len(phi_s) == 0:
            return np.zeros((self.actions_num))
        if self._phi_sa_cache.shape != (self.actions_num, self.features_num):
            self._phi_sa_cache = np.empty((self.actions_num, self.features_num))
        Q = np.multiply(self.weight, phi_s, out=self._phi_sa_cache).sum(axis=1)
        # stacks phi_s in cache
        return Q

    def best_actions(self, s, terminal, p_actions, phi_s=None):
        """
        Returns a list of the best actions at a given state.
        If *phi_s* [the feature vector at state *s*] is given, it is used to
        speed up code by preventing re-computation within this function.

        See :py:meth:`~rlpy.representations.representation.best_action`

        :param s: The given state
        :param terminal: Whether or not the state *s* is a terminal one.
        :param phi_s: (optional) the feature vector at state (s).
        :return: A list of the best actions at the given state.

        """
        Qs = self.Qs(s, terminal, phi_s)
        Qs = Qs[p_actions]
        # Find the index of best actions
        ind = findElemArray1D(Qs, Qs.max())
        return np.array(p_actions)[ind]
