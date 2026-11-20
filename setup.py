"""Build the standalone CSF extension using C++17 and vendored pybind11."""
import os
from pathlib import Path
import shlex
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
NATIVE_SOURCE = ROOT/'scsf/src'
HEADERS = (sorted(NATIVE_SOURCE.glob('*.h')) +
           sorted((ROOT/'deps/pybind11/include').rglob('*.h')))


setup(ext_modules=[Extension('scsf._csf',
    sources=[str(p.relative_to(ROOT)) for p in sorted(NATIVE_SOURCE.glob('*.cpp'))],
    depends=[str(p.relative_to(ROOT)) for p in HEADERS],
    include_dirs=['deps/pybind11/include'],
    libraries=['pthread','m','dl'],
    extra_compile_args=['-std=c++17','-O3','-DNDEBUG','-fvisibility=hidden','-pthread']+
        shlex.split(os.environ.get('EXTRA_CXXFLAGS','')),
    extra_link_args=['-pthread'], language='c++')])
