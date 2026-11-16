"""Shared fixtures for SCSF regression tests."""
from pathlib import Path


def datafile(name):
    return str(Path(__file__).parent/'data'/name)


def load_hamiltonian(filename):
    import numpy as np
    from scsf import hamiltonian
    with np.load(filename, allow_pickle=False) as data:
        return hamiltonian(float(data['ecore']), data['one'], data['two'])
