"""Incremental occupation hashes are filters, including under forced collisions."""
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf import _csf
from scsf.test.test_csf import random_hamiltonian


@pytest.mark.parametrize('nelec,two_s', [(0, 0), (5, 1), (6, 0)])
@pytest.mark.parametrize('bits', [0, 64])
def test_hash_neighbors_equal_exhaustive_distance_even_with_collisions(nelec, two_s, bits):
    w = scsf.csf_wfn(6, nelec, two_s);w.add_all_csfs()
    basis = w._native_basis('packed')
    expected, _, _, _ = _csf._configuration_index_probe(basis, 0, 0)
    actual, stats, _, _ = _csf._configuration_index_probe(basis, 3, 2**20, bits)
    assert actual == expected
    assert stats['mode'] == 3 and 0 < stats['bytes'] <= 2**20
    if bits == 0 and nelec >= 5:
        assert stats['hash_collision_rejections'] > 0


def test_hashes_wide_occupations_and_exact_budget_fallback():
    w = scsf.csf_wfn(130, 6, 0)
    rng = np.random.default_rng(501)
    for _ in range(24):
        label = ['0']*130
        for p in rng.choice([0, 1, 31, 32, 63, 64, 96, 129], 3, replace=False):
            label[p] = '2'
        w.add_csf(''.join(label))
    basis = w._native_basis('packed')
    expected, _, _, _ = _csf._configuration_index_probe(basis, 0, 0)
    actual, stats, _, _ = _csf._configuration_index_probe(basis, 3, 2**20)
    assert actual == expected
    for cap in (0, 1, stats['bytes']-1, stats['bytes']):
        for bits in (0, 64):
            actual, info, _, _ = _csf._configuration_index_probe(basis, 3, cap, bits)
            assert actual == expected and info['bytes'] <= cap
            assert info['mode'] == (3 if cap == stats['bytes'] else 0)




@pytest.mark.parametrize('symmetric', [False, True])
def test_zobrist_stored_matrix_shares_exact_index(symmetric):
    ham = random_hamiltonian(5, seed=16)
    w = scsf.csf_wfn(5, 4, 0);w.add_all_csfs()
    backend = scsf.CSFSpinBackend(ham, matrix_index='zobrist', nthreads=3)
    op = scsf.csf_op(ham, w, backend=backend, symmetric=symmetric, matrix_cutoff=0.)
    ref = scsf.csf_op(ham, w, backend=scsf.CSFReferenceBackend(ham), matrix_cutoff=0.)
    npt.assert_allclose(op.to_scipy().toarray(), ref.to_scipy().toarray(), atol=3e-12)
    assert backend.last_matrix_stats['matrix_index'] == 3


@pytest.mark.parametrize('nrow,ncol', [(31, 17), (17, 31)])
def test_zobrist_rectangular_rows_outside_column_space(nrow, ncol):
    ham = random_hamiltonian(5, seed=76)
    full = scsf.csf_wfn(5, 4, 0);full.add_all_csfs()
    w = scsf.csf_wfn(5, 4, 0)
    for i in np.random.default_rng(66).permutation(len(full)):w.add_csf(full[int(i)])
    backend = scsf.CSFSpinBackend(ham, matrix_index='zobrist', nthreads=3)
    op = scsf.csf_op(ham, w, nrow=nrow, ncol=ncol, symmetric=False,
                     backend=backend, matrix_cutoff=0.)
    reference = scsf.csf_op(ham, w, backend=scsf.CSFReferenceBackend(ham), matrix_cutoff=0.)
    npt.assert_allclose(op.to_scipy().toarray(),
                        reference.to_scipy().toarray()[:nrow, :ncol], atol=3e-12)
    assert backend.last_matrix_stats['matrix_index'] == 3
