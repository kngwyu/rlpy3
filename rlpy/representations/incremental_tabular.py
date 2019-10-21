"""Incrementally expanded Tabular Representation"""
from .representation import Enumerable, Representation
import numpy as np
from copy import deepcopy

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


class IncrementalTabular(Representation, Enumerable):
    """
    Identical to Tabular representation (ie assigns a binary feature function
    f_{d}() to each possible discrete state *d* in the domain, with
    f_{d}(s) = 1 when d=s, 0 elsewhere.
    HOWEVER, unlike *Tabular*, feature functions are only created for *s* which
    have been encountered in the domain, not instantiated for every single
    state at the outset.
    """

    IS_DYNAMIC = True

    def __init__(self, domain, discretization=20):
        self.state_ids = {}
        super().__init__(domain, 0, discretization)

    def phi_non_terminal(self, s):
        hash_key = self._hash_state(s)
        state_id = self.state_ids.get(hash_key)
        if state_id is None:
            self._add_state(s)
            state_id = self.features_num - 1
        F_s = np.zeros(self.features_num, bool)
        F_s[state_id] = 1
        return F_s

    def state_id(self, s):
        hash_id = self._hash_state(s)
        return self.hash.get(hash_id)

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return self._add_state(s) + self._add_state(sn)

    def _add_state(self, s):
        """
        :param s: the (possibly un-cached) state to hash.

        Accepts state ``s``; if it has been cached already, do nothing and
        return 0; if not, add it to the hash table and return 1.
        """

        hash_key = self._hash_state(s)
        if hash_key not in self.state_ids:
            # Assign a new id
            self.state_ids[hash_key] = self.features_num
            # Increment state count
            self.features_num += 1
            # Add a new element to the feature weight vector, theta
            self.add_new_weight()
            return 1
        return 0

    def __deepcopy__(self, memo):
        new_copy = IncrementalTabular(self.domain, self.discretization)
        new_copy.hash = deepcopy(self.hash)
        return new_copy

    def feature_type(self):
        return bool
