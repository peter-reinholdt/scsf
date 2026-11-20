"""SCSF: spin-adapted configuration interaction with native CG and GUGA backends."""
from .integrals import CholeskyHamiltonian, Hamiltonian, PackedHamiltonian, THCHamiltonian, hamiltonian
from .config import get_num_threads, set_num_threads
from .csf import (csf_wfn, csf_dimension, csf_one_electron_product, add_csf_one_electron)
from .csf_structured import csf_structured_op

__version__ = '0.1.0'
__all__ = ['CholeskyHamiltonian', 'Hamiltonian', 'PackedHamiltonian', 'THCHamiltonian',
           'hamiltonian', 'csf_wfn', 'csf_structured_op',
           'csf_dimension', 'csf_one_electron_product', 'add_csf_one_electron',
           'get_num_threads', 'set_num_threads', '__version__']
