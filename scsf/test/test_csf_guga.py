from scsf.test import load_hamiltonian
# This file is part of SCSF, licensed under the GNU GPL version 3 or later.
# See LICENSE for details.
"""GUGA phases, complete generator products, and variational SCI contracts."""
import copy
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

import scsf
from scsf.drivers.solvers import solve_ci
from scsf.test import datafile
from scsf.test.test_csf import determinant_keys, fock_action, random_hamiltonian


def fock_operators(wfn):
    """Independent explicit spin-orbital actions in the existing CG convention."""
    dwfn, transform = wfn.determinant_transform()
    u = transform.toarray()
    dets = [tuple(p for p in range(2*wfn.nbasis) if key & (1 << p))
            for key in determinant_keys(dwfn)]
    lookup = {det: i for i, det in enumerate(dets)}

    def matrix(indices):
        k = wfn.nbasis
        result = np.zeros((len(dets), len(dets)))
        for j, det in enumerate(dets):
            if len(indices) == 2:
                p, q = indices
                actions = [((q+spin*k,), (p+spin*k,)) for spin in range(2)]
            else:
                p, q, r, s = indices
                actions = [((q+sigma*k, s+tau*k), (r+tau*k, p+sigma*k))
                           for sigma, tau in product(range(2), repeat=2)]
            for annihilate, create in actions:
                target, sign = fock_action(det, annihilate, create)
                if sign:
                    result[lookup[target], j] += sign
        return u.T @ result @ u

    return matrix


@pytest.mark.parametrize('k,n,s', [(3,2,0), (3,3,1), (3,4,0), (4,4,0), (4,4,2)])
def test_all_signed_generators_and_four_index_operators(k, n, s):
    wfn = scsf.csf_wfn(k,n,s)
    wfn.add_all_csfs()
    backend = scsf.CSFSpinBackend(random_hamiltonian(k), coupling_engine='guga')
    reference = fock_operators(wfn)
    for indices in product(range(k), repeat=2):
        actual = backend.generator_block(wfn,wfn,s,*indices)
        npt.assert_allclose(actual,reference(indices),rtol=0,atol=3e-14)
        npt.assert_array_equal(actual,backend.generator_block(wfn,wfn,s,*indices[::-1]).T)
    # Exhaust diagonal, coincident, repeated, crossed, nested, and disjoint
    # indices, without relying on spatial-integral symmetries or cancellations.
    for indices in product(range(k), repeat=4):
        actual = backend.two_body_block(wfn,wfn,s,*indices)
        npt.assert_allclose(actual,reference(indices),rtol=0,atol=3e-14)


@pytest.mark.parametrize('k,n,s', [(0,0,0), (2,2,0), (3,3,1), (3,4,0),
                                   (4,4,0), (4,4,2), (4,4,4), (4,8,0)])
def test_complete_guga_hamiltonian_sectors(k, n, s):
    ham = random_hamiltonian(k,seed=31)
    wfn = scsf.csf_wfn(k,n,s)
    wfn.add_all_csfs()
    backend = scsf.CSFSpinBackend(ham,coupling_engine='guga')
    expected = scsf.CSFReferenceBackend(ham).block(wfn,wfn,s)
    actual = backend.block(wfn,wfn,s)
    npt.assert_allclose(actual,expected,rtol=0,atol=6e-14)
    npt.assert_allclose(actual,actual.T,rtol=0,atol=3e-14)
    op = scsf.csf_op(ham,wfn,backend=backend)
    npt.assert_allclose(op.to_scipy().toarray(),expected,rtol=0,atol=6e-14)


def test_intermediate_walk_is_outside_selected_basis():
    backend = scsf.CSFSpinBackend(random_hamiltonian(2),coupling_engine='guga')
    # E_10 sends this singlet to |02>, which is absent from both arguments.
    # The complete product gives 2 and its normal-order correction is 1.
    selected = ['ud']
    assert backend.generator_block(selected,selected,0,0,1)[0,0] == 0
    assert backend.generator_block(selected,selected,0,1,0)[0,0] == 0
    npt.assert_allclose(backend.two_body_block(selected,selected,0,0,1,1,0),[[1.]],atol=1e-15)


@pytest.mark.parametrize('s', [0,2,4])
def test_long_spectator_intervals_and_open_shells(s):
    ham = random_hamiltonian(18,seed=71)
    compact = scsf.csf_wfn(8,8,s)
    compact.add_configuration([1]*8)
    wfn = scsf.csf_wfn(18,10,s)
    for p in (1,16):
        for label in compact[:5]:
            padded = ['0']*18
            padded[p] = '2'
            for site, c in zip((3,5,6,8,10,11,12,14),label):
                padded[site] = c
            wfn.add_csf(''.join(padded))
    guga = scsf.CSFSpinBackend(ham,coupling_engine='guga')
    cg = scsf.CSFSpinBackend(ham,coupling_engine='compiled')
    npt.assert_allclose(guga.block(wfn,wfn,s),cg.block(wfn,wfn,s),rtol=0,atol=3e-13)


def test_twenty_open_shells_without_magnetic_projection_bonds():
    ham = random_hamiltonian(20,seed=18)
    labels = ['u'*10+'d'*10, 'ud'*10, 'uudd'*5]
    backend = scsf.CSFSpinBackend(ham,coupling_engine='guga',max_bond_elements=2)
    actual = backend.block(labels,labels,0)
    expected = scsf.CSFSpinBackend(ham,coupling_engine='compiled').block(labels,labels,0)
    npt.assert_allclose(actual,expected,rtol=0,atol=2e-12)


@pytest.mark.parametrize('workers,cache_bytes', [(1,0), (4,4096), (4,2**20)])
@pytest.mark.parametrize('eps', [0., .2, 2.])
def test_selection_bounds_append_and_workers(workers,cache_bytes,eps):
    ham = random_hamiltonian(5,seed=43)
    base = scsf.csf_wfn(5,4,0)
    for label in ('22000','uudd0','0ud20'):
        base.add_csf(label)
    results=[]
    for engine,screening,implementation in (('guga',True,'native'),
                                          ('guga',False,'native'),
                                          ('guga',False,'python'),
                                          ('compiled',False,'native')):
        wfn=copy.copy(base)
        backend=scsf.CSFSpinBackend(ham,coupling_engine=engine,nthreads=workers,cache_bytes=cache_bytes)
        op=scsf.csf_op(ham,wfn,backend=backend)
        old=op.data()
        scsf.add_csf_hci(ham,wfn,[.6,-.3,.1],eps=eps,backend=backend,
                         screening=screening,implementation=implementation)
        op.update(ham,wfn)
        npt.assert_array_equal(op.data()[:len(old)],old)
        fresh=scsf.csf_op(ham,wfn,backend=backend)
        npt.assert_array_equal(op.data(),fresh.data())
        assert backend.cache_info['bytes'] <= cache_bytes
        results.append((tuple(wfn),op.to_scipy().toarray()))
    assert all(result[0]==results[0][0] for result in results)
    for result in results[1:3]:
        npt.assert_array_equal(result[1],results[0][1])
    for result in results[1:]:
        npt.assert_allclose(result[1],results[0][1],rtol=0,atol=6e-14)
    backend=scsf.CSFSpinBackend(ham,coupling_engine='guga',cache_bytes=cache_bytes)
    full=scsf.csf_wfn(5,4,0)
    full.add_all_csfs()
    matrix=backend.block(full,full,0)
    configs=full.to_configuration_array()
    for i,a in enumerate(configs):
        for j,b in enumerate(configs):
            assert abs(matrix[i,j]) <= backend.configuration_bound(a,b)


@pytest.mark.parametrize('limit', ['max_tensor_entries','max_bond_elements','max_connection_terms',
                                   'max_candidates','max_csfs','max_configurations'])
def test_guga_failed_selection_is_atomic_and_retryable(limit):
    ham=random_hamiltonian(5)
    wfn=scsf.csf_wfn(5,4,0)
    wfn.add_csf('22000')
    backend=scsf.CSFSpinBackend(ham,coupling_engine='guga',nthreads=4)
    kwargs={}
    if limit in ('max_tensor_entries','max_bond_elements','max_connection_terms'):
        setattr(backend,limit,1)
    else:
        kwargs[limit]=1
    with pytest.raises(MemoryError):
        scsf.add_csf_hci(ham,wfn,[1.],eps=0,backend=backend,screening=False,**kwargs)
    assert tuple(wfn)==('22000',) and backend.last_selection_stats=={}
    backend.max_tensor_entries=backend.max_bond_elements=backend.max_connection_terms=1000000
    assert scsf.add_csf_hci(ham,wfn,[1.],eps=0,backend=backend,screening=False)>0


@pytest.mark.parametrize('s', [0,2])
def test_dalton_fixture_and_external_davidson(s):
    ham=load_hamiltonian(datafile('h4_dalton_sto3g.npz'))
    wfn=scsf.csf_wfn(4,4,s)
    wfn.add_all_csfs()
    backend=scsf.CSFSpinBackend(ham,coupling_engine='guga',nthreads=4)
    op=scsf.csf_op(ham,wfn,backend=backend)
    e,c,converged=solve_ci(op.matmat,op.diagonal(),roots=3,tol=1e-11)
    assert converged
    reference=scsf.CSFReferenceBackend(ham).block(wfn,wfn,s)
    npt.assert_allclose(e,np.linalg.eigvalsh(reference)[:3],rtol=0,atol=3e-13)
    npt.assert_allclose(op.matmat(c),c*e,rtol=0,atol=1e-11)


def test_generator_validation_budgets_and_engine_isolation():
    ham=random_hamiltonian(4)
    backend=scsf.CSFSpinBackend(ham,coupling_engine='guga')
    for p,q in ((-1,0),(0,4)):
        with pytest.raises(ValueError,match='index'):
            backend.generator_block(['2200'],['2200'],0,p,q)
    with pytest.raises(ValueError):
        backend.generator_block(['u000'],['2200'],0,0,1)
    backend.max_tensor_entries=1
    with pytest.raises(MemoryError):
        backend.generator_block(['2200'],['2200'],0,0,1)
    backend.max_tensor_entries=1000000
    with pytest.raises(ValueError,match='engine'):
        backend._native.configuration_bound(backend.one,backend.two,[2,2,0,0],[2,2,0,0],
                                             cache=backend._cache,guga=False)


@pytest.mark.parametrize('prescreen,screening', [(True,True),(False,True),(True,False)])
def test_strict_thresholds_and_subnormal_one_body(prescreen,screening):
    for coupling in (.125,np.nextafter(0.,1.)):
        h=np.array([[0.,coupling],[coupling,0.]])
        ham=scsf.hamiltonian(0.,h,np.zeros((2,)*4))
        backend=scsf.CSFSpinBackend(ham,coupling_engine='guga',nthreads=4,prescreen=prescreen)
        assert backend.configuration_bound([0,1],[1,0]) >= coupling
        for eps in (0.,coupling,np.nextafter(coupling,0.),np.nextafter(coupling,1.)):
            wfn=scsf.csf_wfn(2,1,1)
            wfn.add_csf('u0')
            assert scsf.add_csf_hci(ham,wfn,[1.],eps=eps,backend=backend,
                                    screening=screening)==int(coupling>eps)


@pytest.mark.parametrize('screening', [True,False])
def test_strict_threshold_for_same_occupation_spin_recoupling(screening):
    ham=random_hamiltonian(4,seed=12)
    backend=scsf.CSFSpinBackend(ham,coupling_engine='guga',nthreads=4)
    coupling=abs(backend.element('udud','uudd',0))
    assert coupling>0
    for eps in (coupling,np.nextafter(coupling,0.),np.nextafter(coupling,np.inf)):
        wfn=scsf.csf_wfn(4,4,0)
        wfn.add_csf('uudd')
        scsf.add_csf_hci(ham,wfn,[1.],eps=eps,backend=backend,screening=screening)
        assert ('udud' in tuple(wfn))==(coupling>eps)


def test_guga_csr_failure_and_cache_hit_guard():
    ham=random_hamiltonian(4)
    wfn=scsf.csf_wfn(4,4,0)
    wfn.add_csf('2200')
    backend=scsf.CSFSpinBackend(ham,coupling_engine='guga',nthreads=4)
    op=scsf.csf_op(ham,wfn,backend=backend)
    old=op.data()
    stats=dict(backend.last_matrix_stats)
    backend.max_connection_terms=1
    with pytest.raises(MemoryError,match='terms'):
        backend.configuration_bound([2,2,0,0],[2,2,0,0])
    backend.max_connection_terms=1000000
    wfn.add_all_csfs()
    op.max_matrix_entries=1
    with pytest.raises(MemoryError):
        op.update(ham,wfn)
    npt.assert_array_equal(op.data(),old)
    assert backend.last_matrix_stats==stats
    op.max_matrix_entries=10000
    op.update(ham,wfn)
    fresh=scsf.csf_op(ham,wfn,backend=backend)
    npt.assert_array_equal(op.data(),fresh.data())
