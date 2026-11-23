"""SCSF: spin-adapted configuration interaction with native CG and GUGA backends."""
from .integrals import (Hamiltonian, PackedHamiltonian, BlockPackedHamiltonian,
                        hamiltonian, load_hamiltonian, save_hamiltonian)
from .symmetry import PointGroup
from .rdm import BlockRDM2
from .config import get_num_threads, set_num_threads
from .csf import (csf_wfn, csf_dimension, csf_one_electron_product, add_csf_one_electron)
from .csf_structured import csf_structured_op

__version__ = '0.1.0'
__all__ = ['Hamiltonian', 'PackedHamiltonian', 'BlockPackedHamiltonian', 'PointGroup', 'BlockRDM2',
           'hamiltonian', 'load_hamiltonian', 'save_hamiltonian', 'csf_wfn', 'csf_structured_op',
           'csf_dimension', 'csf_one_electron_product', 'add_csf_one_electron',
           'get_num_threads', 'set_num_threads', '__version__']
