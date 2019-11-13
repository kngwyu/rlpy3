"""General Tools for use throughout RLPy"""
import contextlib
import datetime
from itertools import combinations, chain
import numpy as np
from scipy import linalg, stats, special
from scipy.sparse import linalg as slinalg
from scipy import sparse as sp
import os
import sys
from time import process_time as clock
import matplotlib as mpl
from matplotlib import cm, colors, lines, rc
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import path as mpath
from mpl_toolkits.mplot3d import axes3d


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

__rlpy_location__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nogui_mode():
    mpl.use("agg")
    plt.ioff()

    def _stub(*args, **kwargs):
        pass

    plt.show = _stub


# Try GUI backend first
try:
    mpl.use("tkAgg")
    plt.ion()
except ImportError:
    nogui_mode()


def discrete_sample(p):
    cp = np.cumsum(p)
    return np.sum(cp <= np.random.rand(1))


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


# if numpy.version.version < '2.6.0': # Missing count_nonzero


def count_nonzero(arr):
    """
    Custom ``nnz()`` method, moves recursively through any sublists within
    *arr*, such that only individual elements are examined. \n
    Some versions of numpy's count_nonzero only strictly compare each element;
    e.g. ``numpy.count_nonzero([[1,2,3,4,5], [6,7,8,9]])`` returns 2, while
    ``tools.count_nonzero([[1,2,3,4,5], [6,7,8,9]])`` returns 9.

    """

    nnz = 0

    # Is this an instance of a matrix? Use inbuilt nonzero() method and count # of indices returned.
    # NOT TESTED with high-dimensional matrices (only 2-dimensional matrices)
    if sp.issparse(arr):
        return arr.getnnz()

    if isinstance(arr, np.matrixlib.defmatrix.matrix):
        # Tuple of length = # dimensions (usu. 2) containing indices of nonzero
        # elements
        nonzero_indices = arr.nonzero()
        # Find # of indices in the vector corresponding to any of the
        # dimensions (all have same length)
        nnz = np.size(nonzero_indices[0])
        return nnz

    if isinstance(arr, np.ndarray):
        # return sum([1 for x in arr.ravel() if x != 0])
        return np.count_nonzero(arr.ravel())

    if isinstance(arr, list):
        for el in arr:
            if isinstance(el, list):
                nnz += np.count_nonzero(el)
            elif el != 0:
                nnz += 1
        return nnz

    print("In tools.py attempted count_nonzero with unsupported type of", type(arr))
    return None


def randint(low, high, m=1, n=1):
    """
    :param low: Lower bound on possible random ints
    :param high: Max possible random int (INCLUSIVE)
    :param m: number of rows in output
    :param n: number of cols in output

    Generates an ``m x n`` whose elements are integers selected uniform random
    in the range [low, high].

    """
    return np.random.randint(low, high + 1, size=(m, n))


def closestDiscretization(s, num_bins, limits):
    """
    :param s: a state.  (possibly multidimensional) ndarray, with dimension d =
        dimensionality of state space.
    :param num_bins: Number of discrete elements in
    :param limits: 2 x d ndarray, where row[0] is a row vector of the lower
        limit of each discrete dimension, and row[1] are corresponding upper
        limits.

    Returns the closest point to the state ``s`` based on the discretization
    defined by the number of bins and limits. \n
    ( equivalent to state2bin(x) / (num_bins-1) * width + limits[0] )

    """
    # width = limits[1]-limits[0]
    # return round((s-limits[0])*num_bins/(width*1.)) / num_bins * width + limits[0]
    return bin2state(state2bin(s, num_bins, limits), num_bins, limits)


def bin2state(bin, num_bins, limits):
    """
    :param bin: index in the discretization
    :param num_bins: the total number of bins in the discretization
    :param limits: 2 x d ndarray, where row[0] is a row vector of the lower
        limit of each discrete dimension, and row[1] are corresponding upper
        limits.

    .. note::

        This is the inverse of state2bin function.

    Given an index ``bin``, the number of the bins ``num_bins``, and the limits
    on a single state dimension, this function returns the corresponding value
    in the middle of the bin (ie, the average of the discretizations around it)

    """
    bin_width = (limits[1] - limits[0]) / num_bins
    return bin * bin_width + bin_width / 2 + limits[0]


def state2bin(s, num_bins, limits):
    """
    :param s: a state.  (possibly multidimensional) ndarray, with dimension d =
        dimensionality of state space.
    :param num_bins: the total number of bins in the discretization
    :param limits: 2 x d ndarray, where row[0] is a row vector of the lower
        limit of each discrete dimension, and row[1] are corresponding upper
        limits.
    Returns the bin number (index) corresponding to state s given a
    discretization num_bins between each column of limits[0] and limits[1].
    The return value has same dimensionality as ``s``. \n
    Note that ``s`` may be continuous. \n
    \n
    Examples: \n
    s = 0, limits = [-1,5], num_bins = 6 => 1 \n
    s = .001, limits = [-1,5], num_bins = 6 => 1 \n
    s = .4, limits = [-.5,.5], num_bins = 3 => 2 \n

    """

    if s == limits[1]:
        return num_bins - 1
    width = limits[1] - limits[0]
    if s > limits[1]:
        print(
            "Tools.py: WARNING: ", s, " > ", limits[1], ". Using the chopped value of s"
        )
        print("Ignoring", limits[1] - s)
        s = limits[1]
    elif s < limits[0]:
        print(
            "Tools.py: WARNING: ", s, " < ", limits[0], ". Using the chopped value of s"
        )
        #        print("WARNING: %s is out of limits of %s . Using the chopped value of s" %(str(s),str(limits)))
        s = limits[0]
    return int((s - limits[0]) * num_bins / (width * 1.0))


def deltaT(start_time):
    """ Returns the time elapsed since ``start_time`` in seconds. """
    return clock() - start_time


def hhmmss(t):
    """
    :param t: time elapsed (in seconds)
    Returns the string representation of ``t`` in format: ``hhmmss``

    """
    return str(datetime.timedelta(seconds=round(t)))


def className(obj):
    """ Return the name of a class as a string. """
    return obj.__class__.__name__


def createColorMaps():
    """
    Create and register the colormaps to be used in domain visualizations.

    """
    cm.register_cmap(
        cmap=colors.ListedColormap(
            ["w", ".75", "xkcd:bright blue", "xkcd:green", "xkcd:scarlet", "k"],
            "GridWorld",
        )
    )
    cm.register_cmap(
        cmap=colors.ListedColormap(
            [
                "xkcd:navy blue",
                "w",
                "xkcd:light grey",
                "xkcd:wintergreen",
                "xkcd:cherry",
                "k",
            ],
            "DeepSea",
        )
    )
    cm.register_cmap(cmap=colors.ListedColormap(["r", "k"], "fiftyChainActions"))
    cm.register_cmap(cmap=colors.ListedColormap(["b", "r"], "FlipBoard"))
    cm.register_cmap(
        cmap=colors.ListedColormap(["w", ".75", "b", "r"], "IntruderMonitoring")
    )
    cm.register_cmap(
        cmap=colors.ListedColormap(
            ["w", "b", "g", "r", "m", (1, 1, 0), "k"], "BlocksWorld"
        )
    )
    cm.register_cmap(cmap=colors.ListedColormap([".5", "k"], "Actions"))
    cm.register_cmap(cmap=make_colormap({0: "r", 1: "w", 2: "g"}), name="ValueFunction")
    cm.register_cmap(
        cmap=make_colormap({0: "xkcd:scarlet", 1: "w", 2: "xkcd:green"}),
        name="ValueFunction-New",
    )
    cm.register_cmap(
        cmap=colors.ListedColormap(["r", "w", "k"], "InvertedPendulumActions")
    )
    cm.register_cmap(cmap=colors.ListedColormap(["r", "w", "k"], "MountainCarActions"))
    cm.register_cmap(cmap=colors.ListedColormap(["r", "w", "k", "b"], "4Actions"))


def make_colormap(colors):
    """
    Define a new color map based on values specified in the dictionary
    colors, where colors[z] is the color that value z should be mapped to,
    with linear interpolation between the given values of z.

    The z values (dictionary keys) are real numbers and the values
    colors[z] can be either an RGB list, e.g. [1,0,0] for red, or an
    html hex string, e.g. "#ff0000" for red.
    """
    from matplotlib import colors as mc

    z = np.sort(list(colors.keys()))
    min_z = min(z)
    x0 = (z - min_z) / (max(z) - min_z)
    rgb = [mc.to_rgb(colors[zi]) for zi in z]
    cmap_dict = dict(
        red=[(x0[i], c[0], c[0]) for i, c in enumerate(rgb)],
        green=[(x0[i], c[1], c[1]) for i, c in enumerate(rgb)],
        blue=[(x0[i], c[2], c[2]) for i, c in enumerate(rgb)],
    )
    mymap = mc.LinearSegmentedColormap("mymap", cmap_dict)
    return mymap


def showcolors(cmap):
    """
    :param cmap: A colormap.
    Debugging tool: displays all possible values of a colormap.

    """
    plt.clf()
    x = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x, x)
    plt.pcolor(X, Y, 0.5 * (X + Y), cmap=cmap, edgecolors="k")
    plt.axis("equal")
    plt.colorbar()
    plt.title("Plot of x+y using colormap")


def schlieren_colormap(color=[0, 0, 0]):
    """
    Creates and returns a colormap suitable for schlieren plots.
    """
    if color == "k":
        color = [0, 0, 0]
    if color == "r":
        color = [1, 0, 0]
    if color == "b":
        color = [0, 0, 1]
    if color == "g":
        color = [0, 0.5, 0]
    if color == "y":
        color = [1, 1, 0]
    color = np.array([1, 1, 1]) - np.array(color)
    s = np.linspace(0, 1, 20)
    colors = {}
    for key in s:
        colors[key] = np.array([1, 1, 1]) - key ** 10 * color
    schlieren_colors = make_colormap(colors)
    return schlieren_colors


def make_amrcolors(nlevels=4):
    """
    :param nlevels: maximum number of AMR levels expected.

    Make lists of colors useful for distinguishing different grids when
    plotting AMR results.

    Returns the tuple (linecolors, bgcolors):\n
        linecolors = list of nlevels colors for grid lines, contour lines. \n
        bgcolors = list of nlevels pale colors for grid background.

    """

    # For 4 or less levels:
    linecolors = ["k", "b", "r", "g"]
    # Set bgcolors to white, then light shades of blue, red, green:
    bgcolors = ["#ffffff", "#ddddff", "#ffdddd", "#ddffdd"]
    # Set bgcolors to light shades of yellow, blue, red, green:
    # bgcolors = ['#ffffdd','#ddddff','#ffdddd','#ddffdd']

    if nlevels > 4:
        linecolors = 4 * linecolors  # now has length 16
        bgcolors = 4 * bgcolors
    if nlevels <= 16:
        linecolors = linecolors[:nlevels]
        bgcolors = bgcolors[:nlevels]
    else:
        print("*** Warning, suggest nlevels <= 16")

    return (linecolors, bgcolors)


def linear_map(x, a, b, A=0, B=1):
    """
    .. warning::

        ``x`` *MUST* be a scalar for truth values to make sense.

    This function takes scalar ``x`` in range [a,b] and linearly maps it to
    the range [A,B].
    Note that ``x`` is truncated to lie in possible boundaries.

    """
    if a == b:
        res = B
    else:
        res = (x - a) / (1.0 * (b - a)) * (B - A) + A
    if res < A:
        res = A
    if res > B:
        res = B
    return res


def l_norm(x, norm=np.inf):
    """ Returns the L infinity norm of a vector """
    return np.linalg.norm(x, norm)


def generalDot(x, y):
    """
    Takes the inner product of the inputs x and y.
    Defined because of inconsistent or confusing definition of the "dot"
    operator for numpy ndarray, matrix, and sparse.matrix.

    """
    if sp.issparse(x):
        # active_indices = x.nonzero()[0].flatten()
        return x.multiply(y).sum()
    else:
        return np.dot(x, y)


def normpdf(x, mu, sigma):
    """ Returns the scalar probability density of Gaussian (mu,sigma) at x. """
    return stats.norm.pdf(x, mu, sigma)


def factorial(x):
    return special.factorial(x)


def nchoosek(n, k):
    """ Returns combination n choose k. """
    return special.comb(n, k)


def findElemArray1D(x, arr):
    """
    :param x: a scalar
    :param arr: a 1-dimensional numpy ndarray

    Returns an array of indices i in arr where x == arr[i]
    or [] if x not in arr.


    """
    res = np.where(arr == x)
    if len(res[0]):
        return res[0].flatten()
    else:
        return []


def findElemArray2D(x, arr2d):
    """
    :param x: a scalar
    :param arr2d: a 2-dimensional numpy ndarray or matrix

    Returns a tuple of arrays (rVec, cVec), where the corresponding elements in
    each are the rows and cols where arr2d[r,c] == x.
    Returns [] if x not in arr2d. \n

    Example: \n
    arr2d = np.array([[1,2],[3,1]]),  x = 1
    findElemArray2D(x, arr2d) --> ([0, 1],  [0, 1]).
    i.e., arr2d[0][0] and arr2d[1][1]  both == x.

    .. note::

        The type of each tuple member is the same as type(arr2d)

    """

    res = np.where(arr2d == x)
    if len(res[0]):
        return res[0].flatten(), res[1].flatten()
    else:
        return [], []


def perms(X):
    """
    :param X: an iterable type (ndarray, matrix, list).
        If a 1-D array, each element e is treated as the number of discrete
        elements to use for permutations, [0, e).
        If a >1-D array, take permutations between the elements themselves
        between dimensions.

    Returns all permutations *in numpy array format*.  For example: \n
    X = [2 3] \n
    res = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2] \n
    X = [[1,3],[2,3]] \n
    res = [[1,2],[1,3],[3,2],[3,3] \n

    """
    allPerms, _ = perms_r(X, perm_sample=np.array([]), allPerms=None, ind=0)

    return allPerms


######################################################


def perms_r(X, perm_sample=np.array([]), allPerms=None, ind=0):
    """ Recursive helper function for perms(). """
    if allPerms is None:
        # Get memory
        if isinstance(X[0], list):
            size = np.prod([len(x) for x in X])
        else:
            size = np.prod(X, dtype=np.int)
        allPerms = np.zeros((size, len(X)))
    if len(X) == 0:
        allPerms[ind, :] = perm_sample
        perm_sample = np.array([])
        ind = ind + 1
    else:
        if isinstance(X[0], list):
            for x in X[0]:
                allPerms, ind = perms_r(
                    X[1:], np.hstack((perm_sample, [x])), allPerms, ind
                )
        else:
            for x in range(X[0]):
                allPerms, ind = perms_r(
                    X[1:], np.hstack((perm_sample, [x])), allPerms, ind
                )
    return allPerms, ind


######################################################


def vec2id2(x, limits):
    """
    :param x: A discrete (multidimensional) quantity (often the state vector)
    :param limits: The limits of the discrete quantity (often statespace_limits)

    Returns a unique id by determining the number of possible values of ``x``
    that lie within ``limits``, and then seeing where this particular value of
    ``x` falls in that spectrum.

    .. warning::

        This function assumes that (elements of) ``x`` takes integer values,
        and that ``limits`` are the lower and upper bounds on ``x``.

    .. note::

        This implementation is half as fast
        as :py:meth:`~rlpy.tools.general_tools.vec2id`.

    """
    if isinstance(x, int):
        return x
    lim_prod = np.cumprod(limits[:-1])
    return x[0] + sum([x_y[0] * x_y[1] for x_y in zip(x[1:], lim_prod)])


def vec2id(x, limits):
    """
    :param x: A discrete (multidimensional) quantity (often the state vector)
    :param limits: The limits of the discrete quantity (often statespace_limits)

    Returns a unique id by determining the number of possible values of ``x``
    that lie within ``limits``, and then seeing where this particular value of
    ``x` falls in that spectrum.

    .. note::

        See :py:meth:`~rlpy.tools.general_tools.id2vec`, the inverse function.

    .. warning::

        This function assumes that (elements of) ``x`` takes integer values,
        and that ``limits`` are the lower and upper bounds on ``x``.
    """
    if isinstance(x, int):
        return x
    _id = 0
    for d in range(len(x) - 1, -1, -1):
        _id *= limits[d]
        _id += x[d]

    return _id


######################################################


def id2vec(_id, limits):
    """
    :param _id: a unique id, presumably generated using ``vec2id()``.
    :param limits: The limits of the discrete quantity (often statespace_limits)

    Returns the vector corresponding to the unique ``_id`` by determining the
    number of possible values of ``x`` that lie within ``limits``, and then
    seeing which particular vector ``x`` lies at the index ``_id``.

     .. note::

        See :py:meth:`~rlpy.tools.general_tools.vec2id`, the inverse function.

    """
    prods = np.cumprod(limits)
    s = [0] * len(limits)
    for d in range(len(prods) - 1, 0, -1):
        #       s[d] = _id / prods[d-1]
        #       _id %= prods[d-1]
        s[d], _id = divmod(_id, prods[d - 1])
    s[0] = _id
    return s


def bound_vec(X, limits):
    """
    :param X: any (multidimensional) iterable type, eg ndarray or list, len = n.
    :param limits: n x 2 iterable type, where limits[i,0] is minimum possible
        value for dimension i, and limits[i,1] is maximum possible.

    Returns ``X ``with any dimensions that lie outside the bounds of ``limits``
    appropriately truncated. \n
    i.e limits[i,0] <= output[i] <= limits[i,1]

    """
    MIN = limits[:, 0]
    MAX = limits[:, 1]
    X = np.vstack((X, MIN))
    X = np.amax(X, axis=0)
    X = np.vstack((X, MAX))
    X = np.amin(X, axis=0)
    return X


def bound(x, m, M=None):
    """
    :param x: scalar

    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range

    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def powerset(iterable, ascending=1):
    """
    :param iterable: an iterable type (list, ndarray)
    :param ascending: (boolean) if true, return powerset in ascending order,
        else return in descending order.

    """
    s = list(iterable)
    if ascending:
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    else:
        return chain.from_iterable(
            combinations(s, r) for r in range(len(s) + 1, -1, -1)
        )


def printClass(obj):
    """ Print class name and all attributes of object ``obj``. """
    print(className(obj))
    print("=======================================")
    for property, value in vars(obj).items():
        print(property, ": ", value)


def add_new_features(weight_vec, new_elem=None):
    """
    :param weight_vec: The weight vector (often feature weights from
        representation) used for s-a pairs
        (i.e, len(weight_vec) = actions_num * numFeats)
    :param actions_num: The total number of possible actions
    :param new_elem: (Optional) The weights associated with each action of the
        feature to  insert (often new_elem = const * np.ones(actions_num, 1)).
        If not specified or = None, assume 0 weight on new features.

    Adds new elements into ``weight_vec`` in the correct location based on
    the number of possible actions.
    [[Since the new element (usually feature) is added for all actions,
    weight_vec should expand by the number of possible actions as for each
    action the feature vector phi(s) is expand by 1 element.]]\n
    Example: \n
    x = [1,2,3,4], a = 2, new_elem = None => [1,2,0,3,4,0] \n
    x = [1,2,3], a = 3, new_elem = [1,1,1] => [1,1,2,1,3,1] \n

    """
    if new_elem is None:
        new_elem = np.zeros((weight_vec.shape[0], 1))
    elif new_elem.ndim == 1:
        new_elem = new_elem.reshape(-1, 1)
    return np.hstack((weight_vec, new_elem))


def solveLinear(A, b):
    """ Solve the linear equation Ax=b. Return tuple (x, time to solve). """
    error = np.inf  # just to be safe, initialize error variable here
    if sp.issparse(A):
        # print 'sparse', type(A)
        start_log_time = clock()
        result = slinalg.spsolve(A, b)
        solve_time = deltaT(start_log_time)
        error = linalg.norm((A * result.reshape(-1, 1) - b.reshape(-1, 1))[0])
        # For extensive comparision of methods refer to InversionComparison.txt
    else:
        # print 'not sparse, type',type(A)
        if sp.issparse(A):
            A = A.todense()
        # Regularize A
        # result = linalg.lstsq(A,b); result = result[0] # Extract just the
        # answer
        start_log_time = clock()
        result = linalg.solve(A, b)
        solve_time = deltaT(start_log_time)

        # use numpy matrix multiplication
        if isinstance(A, np.matrixlib.defmatrix.matrix):
            error = np.linalg.norm((A * result.reshape(-1, 1) - b.reshape(-1, 1))[0])
        elif isinstance(A, np.ndarray):  # use array multiplication
            error = np.linalg.norm(
                (np.dot(A, result.reshape(-1, 1)) - b.reshape(-1, 1))[0]
            )
        else:
            print(
                "Attempted to solve linear equation Ax=b in solveLinear() of Tools.py with a non-numpy (array / matrix) type."
            )
            sys.exit(1)

    if error > RESEDUAL_THRESHOLD:
        print("||Ax-b|| = %0.1f" % error)
    return result.ravel(), solve_time


def rank(A, eps=1e-12):
    """
    :param A: numpy arrayLike (ndarray, matrix).
    :param eps: threshold above which a singular value is considered nonzero.

    Returns the rank of matrix ``A``, ie number of eigenvalues > ``eps``.
    """
    u, s, v = linalg.svd(A)
    return len([x for x in s if abs(x) > eps])


def fromAtoB(
    x1,
    y1,
    x2,
    y2,
    color="k",
    connectionstyle="arc3,rad=-0.4",
    shrinkA=10,
    shrinkB=10,
    arrowstyle="fancy",
    ax=None,
):
    """
    Draws an arrow from point A=(x1,y1) to point B=(x2,y2) on the (optional)
    axis ``ax``.

    .. note::

        See matplotlib documentation.

    """
    if ax is None:
        return pl.annotate(
            "",
            xy=(x2, y2),
            xycoords="data",
            xytext=(x1, y1),
            textcoords="data",
            arrowprops=dict(
                arrowstyle=arrowstyle,  # linestyle="dashed",
                color=color,
                shrinkA=shrinkA,
                shrinkB=shrinkB,
                patchA=None,
                patchB=None,
                connectionstyle=connectionstyle,
            ),
        )
    else:
        return ax.annotate(
            "",
            xy=(x2, y2),
            xycoords="data",
            xytext=(x1, y1),
            textcoords="data",
            arrowprops=dict(
                arrowstyle=arrowstyle,  # linestyle="dashed",
                color=color,
                shrinkA=shrinkA,
                shrinkB=shrinkB,
                patchA=None,
                patchB=None,
                connectionstyle=connectionstyle,
            ),
        )


def drawHist(data, bins=50, fig=101):
    """
    :param data: Data to use in histogram.
    :param bins: number of bins to use in histogram
    :param fig: The figure number for the plot

    Draws a histogram in its own figure using specified parameters.

    """
    hist, bins = np.histogram(data, bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.figure(fig)
    plt.bar(center, hist, align="center", width=width)


def nonZeroIndex(arr):
    """
    :param arr: a numpy 1-D array.

    Returns the list of indices of nonzero elements in ``arr``. \n
    Example: [0,0,0,1] => [4]

    """
    return arr.nonzero()[0]


def sp_matrix(m, n=1, dtype="float"):
    """
    :param m: number of rows in matrix
    :param n: number of cols in matrix
    :param dtype: datatype of sparse matrix

    Returns an empty sparse matrix with m rows and n columns, with the dtype.

    """
    return sp.csr_matrix((m, n), dtype=dtype)


def sp_dot_array(sp_m, arr):
    """
    :param sp_m: a sparse 1-D array/matrix (created
        with :py:meth:`~rlpy.tools.general_tools.sp_matrix`)
    :param arr: a (possibly dense) 1-D iterable type (ndarray, list, matrix)

    Returns dot product of 1-by-p matrix ``sp_m`` and length-p array arr.

    """
    assert sp_m.shape[1] == len(arr)
    ind = sp_m.nonzero()[1]
    if len(ind) == 0:
        return 0
    if sp_m.dtype == "bool":
        # Just sum the corresponding indexes of theta
        return sum(arr[ind])
    else:
        # Multiply by feature values since they are not binary

        return sum([arr[i] * sp_m[0, i] for i in ind])


def sp_dot_sp(sp_1, sp_2):
    """
    :param sp_1: a sparse 1-D array/matrix (created
        with :py:meth:`~rlpy.tools.general_tools.sp_matrix`)
    :param sp_2: another sparse 1-D array/matrix, len(sp_2) = len(sp_1).

    Returns dot product of 1-by-p matrices ``sp_1`` and ``sp_2``.

    """
    assert sp_1.shape[0] == sp_2.shape[0] and sp_1.shape[1] == 1 and sp_2.shape[1] == 1
    ind_1 = sp_1.nonzero()[0]
    ind_2 = sp_2.nonzero()[0]
    if len(ind_1) * len(ind_2) == 0:
        return 0

    ind = np.intersect1d(ind_1, ind_2)
    # See if they are boolean
    if sp_1.dtype == bool and sp_2.dtype == bool:
        return len(ind)
    sp_bool = None
    if sp_1.dtype == bool:
        sp_bool = sp_1
        sp = sp_2
    if sp_2.dtype == bool:
        sp_bool = sp_2
        sp = sp_1
    if sp_bool is None:
        # Multiply by feature values since they are not binary
        return sum([sp_1[i, 0] * sp_2[i, 0] for i in ind])
    else:
        return sum([sp[i, 0] for i in ind])


def sp_add2_array(sp, arr):
    """
    :param sp: sparse matrix p-by-1 (created
        with :py:meth:`~rlpy.tools.general_tools.sp_matrix`)
    :param arr: a 1-D iterable type (ndarray, list, matrix) of length p.

    Returns ret = arr + sp (with type(ret) = type(arr))

    """
    ind = sp.nonzero()[0]
    for i in ind:
        arr[i] += sp[i, 0]
    return arr


def checkNCreateDirectory(fullfilename):
    """
    :param fullfilename: root path to desired file/folder.

    See if all directories in ``fullfilename`` exist; if not create as required.

    """
    path_, _, _ = fullfilename.rpartition("/")
    if not os.path.exists(path_):
        os.makedirs(path_)


def hasFunction(object, methodname):
    """ Test if class of ``object`` has a method called ``methodname``. """
    method = getattr(object, methodname, None)
    return callable(method)


def pretty(X, format="%0.3f"):
    """
    Returns a formatted string for a numpy array ``X``. \n
    Example: [1,2,3], %0.3f => 1.000    2.000    3.000

    """
    format = format + "\t"
    return "".join(format % x for x in X)


def regularize(A):
    """ Regularize the numpy arrayLike object ``A``.
    Adds REGULARIZATION*I To A, where I is identity matrix and REGULARIZATION
    is defined in general_tools.py.\n
    This is often done before calling the linearSolver.

    .. note::

        ``A`` must be a square matrix.

    """
    x, y = A.shape
    assert x == y  # Square matrix
    if sp.issparse(A):
        A = A + REGULARIZATION * sp.eye(x, x)
        # print 'REGULARIZE', type(A)
    else:
        # print 'REGULARIZE', type(A)
        for i in range(x):
            A[i, i] += REGULARIZATION
    return A


def sparsity(A):
    """ Returns the percentage of nonzero elements in ``A``. """
    return (1 - np.count_nonzero(A) / np.prod(A.shape)) * 100


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


# matplotlib configs
createColorMaps()

DEFAULT_FONTS = {
    "weight": "normal",
    "size": 14,
    "sans-serif": ["Source Han Sans JP", "Source Sans Pro", "Helvetica", "DejaVu Sans"],
}
rc("font", **DEFAULT_FONTS)
rc("pdf", fonttype=42)


@contextlib.contextmanager
def with_pdf_fonts():
    rc("font", **{"sans-serif": ["Dejavu Sans"]})
    yield
    rc("font", **DEFAULT_FONTS)


@contextlib.contextmanager
def with_bold_fonts():
    rc("font", weight="bold")
    yield
    rc("font", weight="normal")


rc("axes", labelsize=12)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
# rc('text',usetex=False)

# Markers
MARKERS = ["o", "s", "D", "^", "*", "x", "p", "+", "v", "|"]
COLOR_LS = [
    [102, 120, 173],
    [118, 167, 125],
    [198, 113, 113],
    [230, 169, 132],
    [169, 193, 213],
    [192, 197, 182],
    [210, 180, 226],
]
COLORS = [[shade / 255.0 for shade in rgb] for rgb in COLOR_LS]

# Colors
PURPLE = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
NOCOLOR = "\033[0m"
RESEDUAL_THRESHOLD = 1e-7
REGULARIZATION = 1e-6
FONTSIZE = 15
SEP_LINE = "=" * 60
