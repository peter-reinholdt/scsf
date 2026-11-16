"""SCSF: spin-adapted configuration interaction with native CG and GUGA backends."""
from .integrals import Hamiltonian, hamiltonian
from .config import get_num_threads, set_num_threads
from .csf import (csf_wfn, csf_op, CSFReferenceBackend, CSFSpinBackend,
                  csf_dimension, add_csf_hci, csf_one_electron_product, add_csf_one_electron)
from .csf_direct import csf_direct_op

__version__ = '0.1.0'
__all__ = ['Hamiltonian', 'hamiltonian', 'csf_wfn', 'csf_op', 'csf_direct_op',
           'CSFReferenceBackend', 'CSFSpinBackend',
           'csf_dimension', 'add_csf_hci', 'csf_one_electron_product', 'add_csf_one_electron',
           'get_num_threads', 'set_num_threads', '__version__']
