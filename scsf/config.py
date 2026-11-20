"""Thread defaults shared by Python CSF APIs; native operators own their workers."""
from operator import index
import os
from pathlib import Path
import subprocess
import sys

_num_threads = 1
_CPU_SYSFS = Path('/sys/devices/system/cpu')


def default_cli_threads():
    """Count available physical cores, falling back to logical CPUs."""
    try:
        available = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        available = None

    if available:
        try:
            # Each SMT sibling reports the same list, even if some siblings
            # are outside this process's affinity mask.
            cores = {(_CPU_SYSFS / f'cpu{cpu}' / 'topology' / 'thread_siblings_list')
                     .read_text().strip() for cpu in available}
        except OSError:
            return min(len(available), 256)
        return min(len(cores), 256)

    if sys.platform == 'darwin':
        try:
            count = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.physicalcpu']))
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
        else:
            if count > 0:
                return min(count, 256)

    cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)
    return min(cpu_count() or 1, 256)


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
