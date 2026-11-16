"""Numerical contracts for every retained direct method and its exact fallbacks."""
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf.test.test_csf import random_hamiltonian, fock_reference, transform_in_order

STRATEGIES = scsf.csf_direct_op._strategies


@pytest.mark.parametrize('strategy', STRATEGIES)
@pytest.mark.parametrize('nelec,two_s,labels', [
    (4, 0, ('uudd', '2200', 'u2d0', 'udud')),
    (4, 2, ('uuud', '2uu0', 'uudu')),
    (3, 1, ('uud0', '2u00', 'u0ud')),
])
def test_direct_selected_hamiltonian_against_fermion_oracle(strategy, nelec, two_s, labels):
    ham = random_hamiltonian(4, seed=950)
    w = scsf.csf_wfn(4, nelec, two_s)
    for label in labels:
        w.add_csf(label)
    # Build H in the complete determinant sector; never project intermediates.
    keys, hdet, _ = fock_reference(ham, nelec, two_s)
    u = transform_in_order(w, keys, two_s)
    matrix = u.T @ hdet @ u
    op = scsf.csf_direct_op(ham, w, strategy=strategy, product_tol=0., nthreads=2,
                            batch_min_pairs=1)
    x = np.random.default_rng(951).normal(size=(len(w), 3))
    x[:, 1] = -x[:, 0]
    x[1:, 2] = 0.
    npt.assert_allclose(op.matmat(x), matrix @ x, rtol=0, atol=3e-12)
    npt.assert_allclose(op.diagonal(), np.diag(matrix), rtol=0, atol=3e-13)
    npt.assert_allclose(op(x[:, 0] + x[:, 2]), matrix @ (x[:, 0] + x[:, 2]), atol=3e-12)
    npt.assert_allclose(x[:, 0] @ op(x[:, 2]), op(x[:, 0]) @ x[:, 2], atol=3e-12)
    npt.assert_array_equal(op(np.zeros(len(w))), 0.)
    assert op.matmat(np.empty((len(w), 0))).shape == (len(w), 0)
    if strategy != 'semi-direct':
        assert op.product_info['hamiltonian_offdiagonal_bytes'] == 0


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_direct_budget_fallbacks_and_transactional_append(strategy):
    ham = random_hamiltonian(5, seed=761)
    full = scsf.csf_wfn(5, 4, 0)
    full.add_all_csfs()
    w = scsf.csf_wfn(5, 4, 0)
    for label in tuple(full)[::4][:3]:
        w.add_csf(label)
    backend = scsf.CSFSpinBackend(ham, matrix_index='zobrist', nthreads=2,
                                 max_index_bytes=0, max_reuse_bytes=0)
    op = scsf.csf_direct_op(ham, w, backend=backend, strategy=strategy, product_tol=0.,
                            max_graph_bytes=0, max_path_bytes=0, max_ordered_bytes=0,
                            max_hamiltonian_bytes=0, max_symbolic_bytes=0, max_trie_bytes=0)
    snapshot = op._native.basis
    for stage in range(2):
        if stage:
            for label in tuple(full)[::4][3:]:
                w.add_csf(label)
            op.update(ham, w)
        reference = scsf.CSFReferenceBackend(ham).block(w, w, 0)
        x = np.random.default_rng(stage).normal(size=(len(w), 2))
        npt.assert_allclose(op.matmat(x), reference @ x, atol=4e-12)
    assert snapshot.labels != tuple(w) and snapshot.context != op.basis_context
    reordered = scsf.csf_wfn(5, 4, 0)
    for label in reversed(tuple(w)):
        reordered.add_csf(label)
    with pytest.raises(ValueError):
        op.update(ham, reordered)
    npt.assert_allclose(op.matmat(x), reference @ x, atol=4e-12)


@pytest.mark.parametrize('strategy', ['compact-spin-guga', 'wide-spin-guga', 'tuga',
                                      'selected-guga', 'hybrid-guga', 'weighted-guga'])
def test_open_shell_fallbacks_and_compressed_paths(strategy):
    # Twelve open shells exceed the tiny memo format. Retain an irregular
    # spin subset, and compare with independent magnetic-projection CG coupling.
    ham = random_hamiltonian(12, seed=314)
    full = scsf.csf_wfn(12, 12, 0)
    full.add_configuration([1] * 12)
    w = scsf.csf_wfn(12, 12, 0)
    for i in (7, 0, 31, 12, 3):
        w.add_csf(full[i])
    reference = scsf.CSFSpinBackend(ham, coupling_engine='compiled').block(w, w, 0)
    op = scsf.csf_direct_op(ham, w, strategy=strategy, product_tol=0.,
                            basis_storage='packed', spin_storage='checkpoints',
                            batch_min_pairs=1, max_batch_elements=32768)
    x = np.random.default_rng(315).normal(size=(len(w), 3))
    npt.assert_allclose(op.matmat(x), reference @ x, atol=2e-11, rtol=2e-13)
    npt.assert_allclose(op.matmat(-x), -reference @ x, atol=2e-11, rtol=2e-13)


def test_above_word_boundary_uses_complete_recurrence():
    k = 65
    one = np.diag(np.arange(k, dtype=float))
    one[0, 64] = one[64, 0] = .2
    ham = scsf.hamiltonian(.3, one, np.zeros((k,) * 4))
    w = scsf.csf_wfn(k, 1, 1)
    w.add_csf('u' + '0' * 64)
    w.add_csf('0' * 64 + 'u')
    op = scsf.csf_direct_op(ham, w, product_tol=0., basis_storage='packed',
                            spin_storage='checkpoints')
    x = np.array([.3, -.7])
    npt.assert_allclose(op(x), one[np.ix_([0, 64], [0, 64])] @ x, atol=1e-13)


@pytest.mark.parametrize('strategy', ['compact-adaptive-guga', 'tuga', 'semi-direct'])
def test_screened_contributions_and_omission_bounds(strategy):
    ham = random_hamiltonian(4)
    w = scsf.csf_wfn(4, 4, 0)
    w.add_all_csfs()
    matrix = scsf.CSFReferenceBackend(ham).block(w, w, 0)
    op = scsf.csf_direct_op(ham, w, strategy=strategy, product_tol=.1, nthreads=2)
    x = np.random.default_rng(942).normal(size=(len(w), 2))
    expected = np.column_stack([
        np.where((np.abs(matrix * v) < .1) & ~np.eye(len(w), dtype=bool),
                 0., matrix * v).sum(axis=1) for v in x.T])
    actual = op.matmat(x)
    npt.assert_allclose(actual, expected, atol=3e-12)
    assert np.all(np.linalg.norm(actual - matrix @ x, axis=0) <= op.last_error_l2 + 3e-12)


@pytest.mark.parametrize('policy', ['budget', 'term'])
def test_generator_bounds_and_unscreened_override(policy):
    ham = random_hamiltonian(4)
    w = scsf.csf_wfn(4, 4, 0)
    w.add_all_csfs()
    matrix = scsf.CSFReferenceBackend(ham).block(w, w, 0)
    op = scsf.csf_direct_op(ham, w, product_tol=.01, generator_tol=.1,
                            generator_screen=policy, batch_min_pairs=1)
    x = np.random.default_rng(446).normal(size=(len(w), 3)) * .01
    actual = op.matmat(x)
    assert np.all(np.linalg.norm(actual - matrix @ x, axis=0) <= op.last_error_l2 + 3e-12)
    npt.assert_allclose(op.matmat(x, unscreened=True), matrix @ x, atol=3e-12)


def test_adaptive_repeated_dense_sparse_blocks_and_output_buffers():
    ham = random_hamiltonian(5)
    w = scsf.csf_wfn(5, 4, 0)
    w.add_all_csfs()
    matrix = scsf.CSFReferenceBackend(ham).block(w, w, 0)
    op = scsf.csf_direct_op(ham, w, strategy='compact-adaptive-guga',
                            product_tol=0., nthreads=3)
    dense = np.random.default_rng(818).normal(size=(len(w), 3))
    sparse = np.zeros(len(w));sparse[0] = 1.
    for x in [dense, sparse, -dense[:, 0], sparse, dense[:, :1]]:
        out = np.empty((len(w), x.shape[1])) if x.ndim == 2 else np.empty(len(w))
        method = op.matmat if x.ndim == 2 else op.matvec
        if x.ndim == 1:
            assert method(x, out=out) is out
        else:
            out = method(x)
        npt.assert_allclose(out, matrix @ x, atol=4e-12)
    for invalid in (np.zeros((len(w) + 1, 1)), np.full((len(w), 1), np.nan)):
        with pytest.raises(ValueError):
            op.matmat(invalid)
    with pytest.raises(TypeError):
        op.matmat(dense.astype(complex))


@pytest.mark.parametrize('cap', [0, 64, 2**20])
def test_semi_direct_value_cache_is_bounded_and_vector_independent(cap):
    ham = random_hamiltonian(4)
    w = scsf.csf_wfn(4, 4, 0)
    w.add_all_csfs()
    matrix = scsf.CSFReferenceBackend(ham).block(w, w, 0)
    op = scsf.csf_direct_op(ham, w, strategy='semi-direct', product_tol=0.,
                            max_hamiltonian_bytes=cap)
    assert op.product_info['hamiltonian_cache_bytes'] <= cap
    for seed in (751, 752):
        x = np.random.default_rng(seed).normal(size=len(w))
        npt.assert_allclose(op(x), matrix @ x, atol=3e-12)


def test_tile_reuse_across_common_closed_and_empty_spectators():
    ham = random_hamiltonian(7, seed=193)
    w = scsf.csf_wfn(7, 6, 0)
    # Identical reduced four-open-shell patterns, different closed-shell
    # positions (including inside operator intervals) and empty spectators.
    for cfg in ([2,1,1,1,1,0,0], [1,2,1,1,1,0,0], [1,1,1,1,2,0,0],
                [0,1,1,2,1,1,0], [1,1,0,1,1,0,2]):
        w.add_configuration(cfg)
    oracle = scsf.csf_op(ham, w, backend=scsf.CSFReferenceBackend(ham), matrix_cutoff=0.)
    op = scsf.csf_direct_op(ham, w, strategy='tuga', product_tol=0.,
        batch_min_pairs=1, max_symbolic_bytes=2**20)
    x = np.random.default_rng(3).normal(size=(len(w), 2))
    npt.assert_allclose(op.matmat(x), oracle.matmat(x), atol=4e-12)
    # First-call hits demonstrate reuse between different physical pairs.
    assert op.last_product_stats['symbolic_hits'] > 0
    npt.assert_allclose(op.matmat(x), oracle.matmat(x), atol=4e-12)


def test_double_transfer_has_complete_intermediate_closure():
    ham = random_hamiltonian(3, seed=913)
    w = scsf.csf_wfn(3, 2, 0)
    for label in ('200', '002'): w.add_csf(label)
    ref = scsf.csf_op(ham, w, backend=scsf.CSFReferenceBackend(ham), matrix_cutoff=0.)
    op = scsf.csf_direct_op(ham, w, strategy='tuga', product_tol=0., batch_min_pairs=1)
    for _ in range(2): npt.assert_allclose(op.matmat(np.eye(2)), ref.to_scipy().toarray(), atol=1e-13)


def test_generator_screening_cannot_poison_unscreened_tiles():
    ham = random_hamiltonian(5)
    w = scsf.csf_wfn(5, 4, 0); w.add_all_csfs()
    op = scsf.csf_direct_op(ham, w, strategy='tuga', product_tol=.01,
        generator_tol=.2, batch_min_pairs=1, max_symbolic_bytes=4096)
    x = np.random.default_rng(89).normal(size=(len(w), 2))
    expected = scsf.csf_op(ham, w, matrix_cutoff=0.).matmat(x)
    y = op.matmat(x)
    assert np.all(np.max(np.abs(y-expected), axis=0) <= op.last_error_inf+4e-12)
    npt.assert_allclose(op.matmat(x, unscreened=True), expected, atol=4e-12)
    y = op.matmat(x)
    assert np.all(np.max(np.abs(y-expected), axis=0) <= op.last_error_inf+4e-12)
