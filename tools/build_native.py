"""Build the optional local float64 sampler: uv run --extra native python tools/build_native.py.

Requires MSVC C++ build tools on Windows, or a system C compiler on Linux/macOS.
No generated binary is required for the reference Python/SciPy fallback.
"""
import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
flags = ['/O2', '/fp:strict'] if os.name == 'nt' else ['-O3', '-fno-fast-math', '-ffp-contract=off']
setup(
    name='steger-precision',
    packages=['steger'],
    package_dir={'': 'src'},
    ext_modules=cythonize(
        [Extension('steger._spline_native', ['src/steger/_spline_native.pyx'], extra_compile_args=flags),
         Extension('steger._mask_native', ['src/steger/_mask_native.pyx'], extra_compile_args=flags)],
        compiler_directives={'language_level': 3},
        build_dir='build/cython',
    ),
    script_args=['build_ext', '--inplace'],
)
