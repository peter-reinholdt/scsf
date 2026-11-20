"""Integral factorization inputs and bounded native contractions."""
import numpy as np
import pytest
import scsf
from .reference.oracle import hamiltonian_block, random_hamiltonian


def test_pair_integrals_on_the_fly_fallback():
    ham = random_hamiltonian(); w = scsf.csf_wfn(4,4,0); w.add_all_csfs()
    full = scsf.csf_structured_op(ham,w,strategy='nminus2')
    info = full.product_info
    assert info['pair_integral_bytes'] > 0
    streamed = scsf.csf_structured_op(ham,w,strategy='nminus2',max_pair_integral_bytes=0)
    assert streamed.product_info['pair_integral_bytes']==0
    x = np.random.default_rng(19).normal(size=(len(w),3))
    np.testing.assert_allclose(streamed.matmat(x),full.matmat(x),atol=1e-12)


@pytest.mark.parametrize('k,n,s,rank', [(4,4,0,5), (4,4,2,5), (4,3,1,5), (3,1,1,0), (0,0,0,0)])
def test_cholesky_products_and_diagonal(k,n,s,rank):
    rng = np.random.default_rng(22)
    factors = rng.normal(size=(rank,k,k))
    factors = (factors+factors.swapaxes(1,2))/2
    one = rng.normal(size=(k,k))
    one = (one+one.T)/2
    ham = scsf.hamiltonian(0.2,one,np.einsum('xpq,xrs->prqs',factors,factors))
    factored = scsf.CholeskyHamiltonian(ham.ecore,one,factors)
    p,q = np.tril_indices(k)
    np.testing.assert_array_equal(factored.pair_factors, factors[:,p,q].T)
    full = scsf.csf_wfn(k,n,s)
    full.add_all_csfs()
    w = scsf.csf_wfn(k,n,s)
    for label in tuple(full)[::-2]:
        w.add_csf(label)
    ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
    x = rng.normal(size=(len(w),3))
    for pair_budget in (None,0):
        op = scsf.csf_structured_op(factored,w,nthreads=2,max_pair_integral_bytes=pair_budget)
        np.testing.assert_allclose(op.matmat(x),ref@x,atol=4e-13,rtol=2e-13)
        np.testing.assert_allclose(op.diagonal(),np.diag(ref),atol=2e-13)
        assert op.ecore == 0.2
        assert op.product_info['eri_factor_rank'] == rank
        assert op.product_info['eri_storage_bytes'] == factored.pair_factors.nbytes
        if pair_budget == 0:
            assert op.product_info['pair_integral_bytes'] == 0
    if k:
        label = next(iter(tuple(full)[1::2]))
        w.add_csf(label)
        op.update(factored,w)
        ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
        np.testing.assert_allclose(op.matmat(np.eye(len(w))),ref,atol=2e-13)


def test_pivoted_cholesky_error_and_input_validation():
    with np.load('scsf/test/data/lih_sto3g.npz',allow_pickle=False) as data:
        ham = scsf.hamiltonian(float(data['ecore']),data['one'],data['two'])
    loose = scsf.CholeskyHamiltonian.from_dense(ham,1e-3)
    tight = scsf.CholeskyHamiltonian.from_dense(ham,1e-10)
    p,q = np.tril_indices(ham.nbasis)
    eri = ham.two_mo[p[:,None],p,q[:,None],q]
    for result,tolerance in [(loose,1e-3),(tight,1e-10)]:
        f = result.pair_factors
        residual = np.max(np.abs(eri-f@f.T))
        assert residual == result.max_eri_error
        assert residual <= tolerance
    assert loose.pair_factors.shape[1] < tight.pair_factors.shape[1]
    one = np.zeros((2,2))
    zero = scsf.CholeskyHamiltonian.from_dense(scsf.hamiltonian(0.,one,np.zeros((2,)*4)))
    assert zero.pair_factors.shape == (3,0)
    bad = scsf.hamiltonian(0.,one,-np.ones((2,)*4))
    with pytest.raises(ValueError,match='cannot be Cholesky factorized'):
        scsf.CholeskyHamiltonian.from_dense(bad)


@pytest.mark.parametrize('strategy,accumulation,threads', [
    ('nminus1','gather',2), ('nminus2','gather',2), ('nminus2','private',1), ('nminus2','private',3),
])
@pytest.mark.parametrize('n,s', [(4,0), (3,1), (4,2)])
def test_packed_integrals_and_streamed_residues(strategy,accumulation,threads,n,s):
    ham = random_hamiltonian()
    packed = scsf.PackedHamiltonian.from_dense(ham)
    full = scsf.csf_wfn(ham.nbasis,n,s)
    full.add_all_csfs()
    w = scsf.csf_wfn(ham.nbasis,n,s)
    for label in tuple(full)[::-2]:
        w.add_csf(label)
    x = np.random.default_rng(11).normal(size=(len(w),4))
    ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
    settings = dict(strategy=strategy,pair_accumulation=accumulation,nthreads=threads,
                    max_pair_integral_bytes=0)
    op = scsf.csf_structured_op(packed,w,**settings)
    tiled = scsf.csf_structured_op(packed,w,**settings,
                                  max_product_scratch_bytes=op.product_info['product_array_bytes_per_rhs'])
    np.testing.assert_allclose(tiled.matmat(x),ref@x,atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(tiled.diagonal(),np.diag(ref),atol=2e-13)
    np.testing.assert_array_equal(tiled.matmat(x),tiled.matmat(x))
    if accumulation == 'private':
        assert op.product_info['product_array_bytes_per_rhs'] == (0 if threads == 1 else 8*threads*len(w))
    w.add_csf(next(label for label in full if label not in tuple(w)))
    op.update(packed,w)
    ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
    np.testing.assert_allclose(op.matmat(np.eye(len(w))),ref,atol=2e-13)


@pytest.mark.parametrize('n,s', [(2,0), (2,2), (3,1), (4,0)])
def test_thc_pair_features_and_uncached_contraction(n,s):
    rng = np.random.default_rng(314+n+s)
    k = 8
    rank = 2 if n==2 else 1
    x = rng.normal(size=(k,rank))
    z = rng.normal(size=(rank,rank))
    z = (z+z.T)*.5  # Includes indefinite cores.
    one = rng.normal(size=(k,k))
    one = (one+one.T)*.5
    ham = scsf.hamiltonian(.1,one,np.einsum('pu,qu,uv,rv,sv->prqs',x,x,z,x,x))
    thc = scsf.THCHamiltonian(ham.ecore,one,x,z)
    full = scsf.csf_wfn(k,n,s)
    full.add_all_csfs()
    w = scsf.csf_wfn(k,n,s)
    for label in tuple(full)[::3]:
        w.add_csf(label)
    ref = hamiltonian_block(ham, tuple(w),tuple(w),s)
    trial = rng.normal(size=(len(w),3))
    for pair_bytes in (None,0):
        op = scsf.csf_structured_op(thc,w,pair_accumulation='private',nthreads=2,
                                   max_pair_integral_bytes=pair_bytes)
        np.testing.assert_allclose(op.matmat(trial),ref@trial,atol=2e-12,rtol=2e-13)
        np.testing.assert_allclose(op.diagonal(),np.diag(ref),atol=2e-13)
        if n==2:
            assert op.product_info['pair_factor_columns'] == (0 if pair_bytes==0 else rank*rank)


def test_collocation_fit_and_dense_thc_pair_cache():
    rng = np.random.default_rng(277)
    k = 5
    x = rng.normal(size=(k,5))
    z = rng.normal(size=(5,5))
    z = (z+z.T)*.5
    two = np.einsum('pu,qu,uv,rv,sv->prqs',x,x,z,x,x)
    ham = scsf.hamiltonian(0.,np.eye(k),two)
    fit = scsf.THCHamiltonian.fit(ham,x,5)
    assert fit.max_eri_error < 1e-12
    w = scsf.csf_wfn(k,3,1)
    w.add_all_csfs()
    op = scsf.csf_structured_op(fit,w,nthreads=2)
    assert op.product_info['pair_integral_bytes'] > 0
    ref = hamiltonian_block(ham, tuple(w),tuple(w),1)
    np.testing.assert_allclose(op.matmat(np.eye(len(w))),ref,atol=2e-12)
