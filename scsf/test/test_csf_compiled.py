# This file is part of SCSF, licensed under the GNU GPL version 3 or later.
# See LICENSE for details.
"""Compiled local maps versus the retained fermionic-string contraction oracle."""
import copy

import numpy as np
import numpy.testing as npt
import pytest

import scsf
from scsf.test.test_csf import random_hamiltonian


@pytest.mark.parametrize('k,n,s', [(0,0,0), (4,4,0), (4,4,2), (4,4,4),
                                  (4,3,1), (4,6,0), (5,4,0)])
def test_compiled_signed_blocks_and_csr_equal_reference(k, n, s):
    ham = random_hamiltonian(k, seed=88)
    wfn = scsf.csf_wfn(k,n,s)
    wfn.add_all_csfs()
    compiled = scsf.CSFSpinBackend(ham, coupling_engine='compiled', nthreads=4)
    reference = scsf.CSFSpinBackend(ham, coupling_engine='python')
    npt.assert_array_equal(compiled.block(wfn,wfn,s), reference.block(wfn,wfn,s))
    a,b = (scsf.csf_op(ham,wfn,backend=backend) for backend in (compiled,reference))
    for name in ('data','indices','indptr'):
        npt.assert_array_equal(getattr(a,name)(),getattr(b,name)())


@pytest.mark.parametrize('s', [0,2])
def test_long_closed_prefixes_and_different_open_site_sets(s):
    ham = random_hamiltonian(18,seed=13)
    compact = scsf.csf_wfn(6,4,s)
    compact.add_all_csfs()
    wfn = scsf.csf_wfn(18,6,s)
    for i in np.random.default_rng(7).choice(len(compact),24,replace=False):
        label = ['0']*18
        label[1] = '2'
        for p,step in zip((3,5,8,9,12,16),compact[int(i)]):
            label[p] = step
        wfn.add_csf(''.join(label))
    a,b = (scsf.CSFSpinBackend(ham,coupling_engine=engine).block(wfn,wfn,s)
           for engine in ('compiled','python'))
    npt.assert_array_equal(a,b)
    npt.assert_allclose(a,scsf.CSFReferenceBackend(ham).block(wfn,wfn,s),rtol=0,atol=2e-13)


def test_many_open_shells_with_nontrivial_magnetic_bonds():
    ham = random_hamiltonian(10,seed=43)
    wfn = scsf.csf_wfn(10,10,0)
    wfn.add_configuration([1]*10)
    a,b = (scsf.CSFSpinBackend(ham,coupling_engine=engine).block(wfn,wfn,0)
           for engine in ('compiled','python'))
    npt.assert_array_equal(a,b)


@pytest.mark.parametrize('eps', [0., .1, 1., 1e6])
def test_compiled_selection_and_atomic_append_equal_reference(eps):
    ham = random_hamiltonian(5)
    base = scsf.csf_wfn(5,4,0)
    base.add_csf('uudd0')
    base.add_csf('22000')
    results = []
    for engine in ('compiled','python'):
        wfn = copy.copy(base)
        backend = scsf.CSFSpinBackend(ham,coupling_engine=engine,nthreads=4,cache_bytes=4096)
        op = scsf.csf_op(ham,wfn,backend=backend)
        old = op.data()
        scsf.add_csf_hci(ham,wfn,[.8,-.2],eps=eps,backend=backend)
        op.update(ham,wfn)
        npt.assert_array_equal(op.data()[:len(old)],old)
        results.append((tuple(wfn),op.data(),op.indices(),op.indptr()))
        assert backend.cache_info['bytes'] <= 4096
    assert results[0][0] == results[1][0]
    for a,b in zip(results[0][1:],results[1][1:]):
        npt.assert_array_equal(a,b)


def test_coupling_engine_validation():
    with pytest.raises(ValueError,match='coupling_engine'):
        scsf.CSFSpinBackend(random_hamiltonian(2),coupling_engine='unknown')


@pytest.mark.parametrize('k', [32, 33])
def test_direct_endpoint_lookup_packed_boundary_and_vector_fallback(k):
    # Enough distinct occupations force endpoint lookup instead of selected-key
    # scanning. Put occupied sites at the end to exercise the high packed bits.
    compact = scsf.csf_wfn(14, 4, 0)
    compact.add_csf('2'+'0'*12+'2')
    compact.add_all_csfs()
    sites = list(range(13))+[k-1]
    wfn = scsf.csf_wfn(k, 4, 0)
    for label in compact:
        padded = ['0']*k
        for p, step in zip(sites, label):
            padded[p] = step
        wfn.add_csf(''.join(padded))
    h = np.random.default_rng(21).normal(size=(14,14))
    h += h.T
    padded_h = np.zeros((k,k))
    padded_h[np.ix_(sites,sites)] = h
    ham = scsf.hamiltonian(0., padded_h, np.zeros((k,)*4))
    backend = scsf.CSFSpinBackend(ham, nthreads=4, coupling_engine='compiled', matrix_index='reference')
    op = scsf.csf_op(ham, wfn, nrow=1, symmetric=False, backend=backend)
    assert backend.last_matrix_stats['configuration_lookups'] > 0
    reference_ham = scsf.hamiltonian(0., h, np.zeros((14,)*4))
    reference = scsf.csf_op(reference_ham, compact, nrow=1, symmetric=False,
                            backend=scsf.CSFReferenceBackend(reference_ham))
    for name in ('data', 'indices', 'indptr'):
        npt.assert_allclose(getattr(op,name)(),getattr(reference,name)(),rtol=0,atol=2e-14)
