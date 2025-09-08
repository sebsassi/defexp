from distutils.core import setup
from distutils.extension import Extension
import numpy
import os

cython_environment_variable = "USE_CYTHON"
USE_CYTHON = bool(int(os.environ.get(cython_environment_variable, None)))

inc_dirs = [numpy.get_include()]

suffix = ".pyx" if USE_CYTHON else ".c"

extensions = [
        Extension(
                "voronoi_occupation",
                [
                    "voronoi_occupation" + suffix,
                    "voronoi_occupation_c.c"],
                include_dirs=inc_dirs,
                extra_compile_args=["-O3", "-Wall", "-Wextra", "-Wconversion", "-Wpedantic"])
              ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)
    print("Using Cython")

setup(ext_modules=extensions)
        
