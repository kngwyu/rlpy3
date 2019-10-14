"""
Installation script for RLPy

Large parts of this file were taken from the pandas project
(https://github.com/pydata/pandas) which have been permitted for use under the
BSD license.
"""
import glob
import io
import os
from os.path import join as pjoin
import re
import shutil
import sys

from distutils.command.build import build
from distutils.command.sdist import sdist
import pkg_resources
from setuptools import Command, Extension, find_packages, setup

try:
    from Cython.Distutils import build_ext as _build_ext

    HAS_CYTHON = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext

    HAS_CYTHON = False


if sys.platform == "darwin":
    # by default use clang++ as this most likely to have c++11 support
    # on OSX
    if "CC" not in os.environ or os.environ["CC"] == "":
        os.environ["CC"] = "clang++"
        extra_args = ["-std=c++0x", "-stdlib=libc++"]
else:
    extra_args = []


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename("numpy", "core/include")

        for ext in self.extensions:
            if hasattr(ext, "include_dirs") and numpy_incl not in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


class CheckingBuildExt(build_ext):
    """Subclass build_ext to get clearer report if Cython is necessary."""

    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    raise Exception(
                        "Cython-generated file '{}' not found."
                        "Cython is required to compile rlpy from a development branch."
                        "Please install Cython or download a release package of rlpy.".format(
                            src
                        )
                    )

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        build_ext.build_extensions(self)


class CythonCommand(build_ext):
    """Custom distutils command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op."""

    def build_extension(self, ext):
        pass


class DummyBuildSrc(Command):
    """ numpy's build_src command interferes with Cython's build_ext.
    """

    user_options = []

    def initialize_options(self):
        self.py_modules_dict = {}

    def finalize_options(self):
        pass

    def run(self):
        pass


class CheckSDist(sdist):
    """Custom sdist that ensures Cython has compiled all pyx files to c."""

    _pyxfiles = [
        "rlpy/representations/hashing.pyx",
        "rlpy/domains/HIVTreatment_dynamics.pyx",
        "rlpy/representations/kernels.pyx",
    ]

    def initialize_options(self):
        sdist.initialize_options(self)

    def run(self):
        if "cython" in CMD_CLASS:
            self.run_command("cython")
        else:
            for pyxfile in self._pyxfiles:
                cfile = pyxfile[:-3] + "c"
                cppfile = pyxfile[:-3] + "cpp"
                msg = (
                    "C-source file '%s' not found." % (cfile)
                    + " Run 'setup.py cython' before sdist."
                )
                assert os.path.isfile(cfile) or os.path.isfile(cppfile), msg
        sdist.run(self)


class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = ["transformations.c"]

        for root, dirs, files in list(os.walk("rlpy")):
            for f in files:
                if f in self._clean_exclude:
                    continue

                if os.path.splitext(f)[-1] in (
                    ".pyc",
                    ".so",
                    ".o",
                    ".pyo",
                    ".pyd",
                    ".c",
                    ".orig",
                ):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == "__pycache__":
                    self._clean_trees.append(pjoin(root, d))

        for d in ("build",):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except Exception:
                pass


CMD_CLASS = {
    "clean": CleanCommand,
    "build": build,
    "build_ext": CheckingBuildExt,
    "sdist": CheckSDist,
}

if HAS_CYTHON:
    CMD_CLASS["cython"] = CythonCommand
else:
    CMD_CLASS["build_src"] = DummyBuildSrc

# always cythonize if C-files are not present
USE_CYTHON = not os.path.exists("rlpy/representations/hashing.c") or os.getenv(
    "USE_CYTHON", False
)
extensions = [
    Extension(
        "rlpy.representations.hashing",
        ["rlpy/representations/hashing.pyx"],
        include_dirs=["rlpy/representations"],
    ),
    Extension(
        "rlpy.domains.HIVTreatment_dynamics",
        ["rlpy/domains/HIVTreatment_dynamics.pyx"],
        include_dirs=["rlpy/representations"],
    ),
    Extension(
        "rlpy.representations.kernels",
        [
            "rlpy/representations/kernels.pyx",
            "rlpy/representations/c_kernels.cc",
            "rlpy/representations/c_kernels.pxd",
        ],
        language="c++",
        include_dirs=["rlpy.representations"],
    ),
    Extension(
        "rlpy.tools._transformations", ["rlpy/tools/transformations.c"], include_dirs=[]
    ),
]


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            elif ext in (".pxd"):
                continue
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


if HAS_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)
else:
    extensions = no_cythonize(extensions)


here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "rlpy/__init__.py"), "rt", encoding="utf8") as f:
    VERSION = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)


with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = "\n" + f.read()


REQUIRES_PYTHON = ">=3.5.0"

REQUIRED = ["numpy>=1.15", "scipy>=1.3", "matplotlib>=3.1", "click>=6.0", "joblib"]

EXTRA = {"systemadmin": "networkx", "bebf": "scikit-learn"}

setup(
    name="rlpy3",
    version=VERSION,
    maintainer="Yuji Kanagawa",
    maintainer_email="yuji.kngw.80s.revive@gmail.com",
    license="BSD 3-clause",
    url="https://github.com/kngwyu/rlpy",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=False,
    cmdclass=CMD_CLASS,
    description="Value-Function-Based Reinforcement-Learning Library for"
    + " Education and Research: Python3 Fork",
    long_description=LONG_DESCRIPTION,
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRA,
    setup_requires=["numpy >= 1.7"],
    ext_modules=extensions,
    test_suite="tests",
)
