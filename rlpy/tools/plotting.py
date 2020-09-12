"""Plotting utilities
"""
import contextlib
import matplotlib as mpl
from matplotlib import cm, colors, lines, rc  # noqa
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches  # noqa
from matplotlib import path as mpath  # noqa
import numpy as np


class ModeHolder:
    def __init__(self):
        self._inner = False

    def __bool__(self):
        return self._inner


JUPYTER_MODE = ModeHolder()


def jupyter_mode(mode=True):
    global JUPYTER_MODE
    JUPYTER_MODE._inner = True


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


def create_color_maps():
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
                "w",
                ".60",
                "xkcd:pale blue",
                "xkcd:pale light green",
                "xkcd:pale red",
                "k",
            ],
            "PaleGridWorld",
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

    return linecolors, bgcolors


def from_a_to_b(
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
        ax = pl.gca()
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


def set_xticks(ax, xticks, labels=None, position=None, fontsize=None):
    if position is not None:
        ax.get_xaxis().set_ticks_position(position)
    ax.set_xticks(xticks)
    if fontsize is not None:
        for l in ax.get_xticklabels():
            l.set_fontsize(fontsize)


def set_yticks(ax, yticks, position=None, fontsize=None):
    if position is not None:
        ax.get_yaxis().set_ticks_position(position)
    ax.set_yticks(yticks)
    if fontsize is not None:
        for l in ax.get_yticklabels():
            l.set_fontsize(fontsize)


# matplotlib configs
create_color_maps()

DEFAULT_FONTS = {
    "weight": "normal",
    "size": 14,
    "sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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


@contextlib.contextmanager
def with_scaled_figure(scale_x, scale_y):
    x, y = plt.rcParams["figure.figsize"]
    rc("figure", figsize=(x * scale_x, y * scale_y))
    yield
    rc("figure", figsize=(x, y))


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
FONTSIZE = 15
SEP_LINE = "=" * 60
