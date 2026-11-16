"""Thread defaults shared by Python CSF APIs; native operators own their workers."""
from operator import index
import os

_num_threads = 1


def set_num_threads(nthreads):
    global _num_threads
    nthreads = index(nthreads)
    if not 1 <= nthreads <= 256:
        raise ValueError('nthreads must be between 1 and 256')
    _num_threads = nthreads


def get_num_threads():
    return _num_threads


if 'SCSF_NUM_THREADS' in os.environ:
    set_num_threads(int(os.environ['SCSF_NUM_THREADS']))
