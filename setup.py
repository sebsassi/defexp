import os
import sys
from setuptools import setup, Extension
import numpy

USE_CYTHON = bool(int(os.environ.get("USE_CYTHON", "0")))

inc_dirs = [numpy.get_include(), "defexp/cinclude"]

suffix = ".pyx" if USE_CYTHON else ".c"

extensions = [
        Extension(
                "voronoi_occupation",
                [
                    "defexp/defexp.voronoi_occupation" + suffix,
                    "defexp/csrc/voronoi_occupation_c.c"],
                include_dirs=inc_dirs,
                extra_compile_args=["-O3", "-Wall", "-Wextra", "-Wconversion", "-Wpedantic"])
              ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)
    print("Using Cython")

setup(ext_modules=extensions)
