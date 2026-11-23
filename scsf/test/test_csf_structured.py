"""Packed integral inputs and bounded native contractions."""
import numpy as np
import pytest
import scsf
from .reference.oracle import hamiltonian_block, random_hamiltonian


@pytest.mark.parametrize('strategy,accumulation,threads', [
    ('nminus1','gather',2), ('nminus2','gather',2), ('nminus2','private',1), ('nminus2','private',3),
])
@pytest.mark.parametrize('n,s', [(4,0), (3,1), (4,2)])
def test_packed_integrals_and_residue_products(strategy,accumulation,threads,n,s):
    ham = random_hamiltonian()
    packed = scsf.PackedHamiltonian.from_dense(ham)
    full = scsf.csf_wfn(ham.nbasis,n,s)
    full.add_all_csfs()
    w = scsf.csf_wfn(ham.nbasis,n,s)
    for label in tuple(full)[::-2]:
        w.add_csf(label)
    x = np.random.default_rng(11).normal(size=(len(w),4))
    ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
    op = scsf.csf_structured_op(packed,w,strategy=strategy,pair_accumulation=accumulation,nthreads=threads)
    np.testing.assert_allclose(op.matmat(x),ref@x,atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(op.diagonal(),np.diag(ref),atol=2e-13)
    if accumulation == 'private':
        assert op.product_info['product_array_bytes_per_rhs'] == (0 if threads == 1 else 8*threads*len(w))
    w.add_csf(next(label for label in full if label not in tuple(w)))
    op.update(packed,w)
    ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
    np.testing.assert_allclose(op.matmat(np.eye(len(w))),ref,atol=2e-13)
