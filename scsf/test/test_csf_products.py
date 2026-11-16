"""Native CSR products, CSR ownership, and transactional SCI updates."""
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np
import numpy.testing as npt
import pytest
from scipy import sparse
import scsf
from scsf import _csf
from scsf.test.test_csf import random_hamiltonian

def native(matrix, symmetric=False, threads=1, mode='symmetric'):
    return _csf._CSRProducts(matrix.data, matrix.indices, matrix.indptr, *matrix.shape, symmetric, threads, mode)

@pytest.mark.parametrize('threads', [1, 3])
@pytest.mark.parametrize('mode', ['native', 'native-gather'])
@pytest.mark.parametrize('shape,symmetric', [((19, 19), True), ((13, 19), False), ((0, 19), False), ((19, 0), False), ((0, 0), True)])
def test_native_sparse_vector_and_block_products(shape, symmetric, threads, mode):
    rng = np.random.default_rng(71)
    a = rng.normal(size=shape)
    a[rng.random(shape) < 0.7] = 0
    if symmetric:
        a = np.tril(a)
    csr = sparse.csr_matrix(a)
    op = native(csr, symmetric, threads, mode)
    full = a + np.tril(a, -1).T if symmetric else a
    x = rng.normal(size=shape[1])
    for v in (x, x[::-1], x.astype(np.float32), list(x)):
        npt.assert_allclose(op.matvec(v), full @ v, rtol=2e-14, atol=2e-14)
    for columns in (0, 1, 2, 8):
        x = rng.normal(size=(shape[1], columns))
        for block in (x, np.asfortranarray(x), x[::-1], x[:, ::-1], x.tolist()):
            if shape[1] == 0 and isinstance(block, list):
                continue
            actual = op.matmat(block)
            assert actual.shape == (shape[0], columns)
            npt.assert_allclose(actual, full @ block, rtol=2e-14, atol=2e-14)
    expected = op.matvec(np.ones(shape[1]))
    csr.data[:] = 999
    csr.indices[:] = -1
    del csr
    gc.collect()
    npt.assert_array_equal(op.matvec(np.ones(shape[1])), expected)

@pytest.mark.parametrize('engine', ['native', 'scipy'])
@pytest.mark.parametrize('mode', ['native', 'native-gather'])
@pytest.mark.parametrize('symmetric', [True, False])
def test_operator_products_append_alias_and_failed_preparation(engine, mode, symmetric, monkeypatch):
    ham = random_hamiltonian(4)
    wfn = scsf.csf_wfn(4, 4, 0)
    wfn.add_csf('2200')
    backend = scsf.CSFSpinBackend(ham, coupling_engine='guga', nthreads=4)
    op = scsf.csf_op(ham, wfn, backend=backend, symmetric=symmetric, product_engine=engine, product_nthreads=2, product_mode=mode, storage_engine='snapshot')
    old_products = op._products
    old_stats = backend.last_matrix_stats.copy()
    old = op.to_scipy()
    wfn.add_all_csfs()

    def fail(matrix):
        raise MemoryError('injected product allocation failure')
    with monkeypatch.context() as patch:
        patch.setattr(op, '_prepare_products', fail)
        with pytest.raises(MemoryError, match='allocation'):
            op.update(ham, wfn)
    assert op._products is old_products and op.shape == old.shape
    assert backend.last_matrix_stats == old_stats
    npt.assert_array_equal(op([2.0]), old @ np.array([2.0]))
    op.update(ham, wfn)
    a = op.to_scipy()
    x = np.arange(len(wfn), dtype=float)
    expected = a @ x
    assert op.matvec(x, out=x) is x
    npt.assert_allclose(x, expected, atol=2e-13)
    out = np.zeros(len(wfn) * 2)[::2]
    assert op.matvec(np.ones(len(wfn)), out=out) is out
    npt.assert_allclose(out, a @ np.ones(len(wfn)), atol=2e-13)
    block = np.arange(len(wfn) * 3).reshape(len(wfn), 3)
    npt.assert_allclose(op.matmat(block), a @ block, atol=2e-12)
    if old_products is not None:
        npt.assert_array_equal(old_products.matvec([2.0]), old @ np.array([2.0]))
        stored = 2 * op.size - len(wfn) if mode == 'native-gather' and symmetric else op.size
        assert op.product_info['csr_bytes'] == stored * 16 + (len(wfn) + 1) * 8
    assert op.product_info['engine'] == engine and op.product_nthreads == 2
    op.data()[:] = -100
    npt.assert_allclose(op.matmat(block), a @ block, atol=2e-12)

@pytest.mark.parametrize('values,indices,ptr,shape,symmetric', [([1.0], [0], [0, 2], (1, 1), True), ([1.0], [0], [1, 1], (1, 1), True), ([1.0], [1], [0, 1], (1, 1), False), ([1.0], [-1], [0, 1], (1, 1), False), ([1.0, 2.0], [0, 0], [0, 2], (1, 1), False), ([1.0, 2.0], [1, 0], [0, 2, 2], (2, 2), False), ([1.0], [1], [0, 1, 1], (2, 2), True), ([np.inf], [0], [0, 1], (1, 1), True), ([1.0], [0], [0, 1, 0, 1], (3, 3), True), ([], [], [], (0, 0), True), ([], [], [0, 0], (1, 2), True)])
def test_native_rejects_invalid_csr(values, indices, ptr, shape, symmetric):
    with pytest.raises(ValueError):
        _csf._CSRProducts(np.array(values, dtype=float), np.array(indices, dtype=np.int64), np.array(ptr, dtype=np.int64), *shape, symmetric)

@pytest.mark.parametrize('mode', ['native', 'native-gather'])
def test_product_validation_and_concurrent_calls(mode):
    ham = random_hamiltonian(2)
    wfn = scsf.csf_wfn(2, 2, 0)
    wfn.add_all_csfs()
    for kwargs in ({'product_engine': 'bad'}, {'product_mode': 'bad'}, {'product_nthreads': 0}, {'product_nthreads': 257}, {'product_nthreads': 1.5}):
        with pytest.raises((ValueError, TypeError)):
            scsf.csf_op(ham, wfn, **kwargs)
    op = scsf.csf_op(ham, wfn, product_nthreads=4, product_mode=mode)
    n = len(wfn)
    x = np.ones((n, 3))
    expected = op.to_scipy() @ x
    with ThreadPoolExecutor(4) as pool:
        for result in pool.map(op.matmat, [x] * 12):
            npt.assert_allclose(result, expected, atol=1e-13)
    op.matmat(x)
    op.matvec(x[:, 0])
    for f, arg in ((op.matvec, np.ones(n + 1)), (op.matmat, np.ones((n + 1, 2))), (op.matmat, np.ones(n)), (op.matvec, np.full(n, np.nan)), (op.matmat, np.full((n, 1), np.inf))):
        with pytest.raises(ValueError):
            f(arg)
    for f, arg in ((op.matvec, np.ones(n, dtype=complex)), (op.matmat, np.ones((n, 1), dtype=complex))):
        with pytest.raises(TypeError):
            f(arg)
    readonly = np.ones(n)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        op.matvec(np.ones(n), out=readonly)
    with pytest.raises(ValueError):
        op._products.matvec(np.ones(n + 1))
    with pytest.raises(ValueError):
        op._products.matmat(np.full((n, 1), np.nan))
    with pytest.raises(TypeError):
        op._products.matvec(np.ones(n, dtype=complex))

@pytest.mark.parametrize('mode', ['native', 'native-gather'])
def test_multiple_partitions_cover_every_row_and_reflected_contribution(mode):
    a = np.tril(np.random.default_rng(8).normal(size=(2003, 2003)))
    a[-2:] = 0
    csr = sparse.csr_matrix(a)
    op = native(csr, True, 12 if mode == 'native-gather' else 4, mode)
    assert op.active_threads == (12 if mode == 'native-gather' else 2)
    if mode == 'native-gather':
        assert op.output_scratch_bytes_per_column == 0
    full = csr + sparse.tril(csr, -1).T
    for columns in (1, 3):
        x = np.random.default_rng(7).normal(size=(2003, columns))
        npt.assert_allclose(op.matmat(x), full @ x, atol=2e-11, rtol=1e-12)


@pytest.mark.parametrize('nelec,two_s,threads', [(4,0,1), (4,2,3), (3,1,2)])
def test_ranked_csr_construction_matches_independent_oracle(nelec, two_s, threads):
    ham = random_hamiltonian(5, seed=412)
    full = scsf.csf_wfn(5, nelec, two_s)
    full.add_all_csfs()
    w = scsf.csf_wfn(5, nelec, two_s)
    for i in np.random.default_rng(413).permutation(len(full))[::3]:
        w.add_csf(full[int(i)])
    expected = scsf.CSFReferenceBackend(ham).block(w, w, two_s)
    backend = scsf.CSFSpinBackend(ham, matrix_kernel='ranked', nthreads=threads,
                                 max_reuse_bytes_per_worker=4096)
    op = scsf.csf_op(ham, w, backend=backend, matrix_cutoff=0.)
    npt.assert_allclose(op.to_scipy().toarray(), expected, atol=3e-12)
