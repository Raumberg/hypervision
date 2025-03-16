from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "aimc",
        ["aimc.pyx"],
        libraries=["user32"],  # Link against Windows user32 library
        extra_compile_args=["/O2"],  # Optimizations for MSVC
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'nonecheck': False
    }),
)