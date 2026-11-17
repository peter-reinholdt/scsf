"""Numerical contracts for every retained direct method and its exact fallbacks."""
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf.test.test_csf import random_hamiltonian, fock_reference, transform_in_order

STRATEGIES = scsf.csf_direct_op._strategies


@pytest.mark.parametrize('tolerance', [0., .1])
def test_parallel_activity_scan_preserves_sparse_dense_and_zero_products(tolerance):
    # More than 4096 configurations exercise coarse parallel scanning. Shuffle
    # CSFs so the configurations own interleaved input/output rows. A diagonal
    # one-body Hamiltonian gives an independent oracle without a dense matrix.
    k = 11
    ham = scsf.hamiltonian(.3, np.diag(np.arange(1., k+1)), np.zeros((k,)*4))
    full = scsf.csf_wfn(k, 6, 0)
    full.add_all_csfs()
    labels = tuple(full)
    rng = np.random.default_rng(81429)
    w = scsf.csf_wfn(k, 6, 0)
    for i in rng.permutation(len(full)):
        w.add_csf(labels[i])
    diagonal = np.array([sum((2 if c == '2' else c in 'ud')*(i+1)
                             for i, c in enumerate(label)) for label in w])
    backend = scsf.CSFSpinBackend(ham, nthreads=4, matrix_index='zobrist',
                                  max_reuse_bytes=0)
    op = scsf.csf_direct_op(ham, w, backend=backend, strategy='compact-adaptive-guga',
                            product_tol=tolerance, max_graph_bytes=0)
    dense = rng.normal(size=(len(w), 3))
    sparse = np.zeros((len(w), 1))
    sparse[[0, 17, 46, 503], 0] = [1., -2., .7, .3]
    for x in (sparse, dense, np.zeros_like(dense), sparse):
        npt.assert_allclose(op.matmat(x), diagonal[:, None]*x, rtol=0, atol=1e-12)
        assert op.last_product_stats['activity_workers'] == 4
        assert op.product_info['product_team_creations'] == 1
        if tolerance == 0. and x is sparse:
            assert op.last_product_stats['source_traversal']
            assert op.last_product_stats['workers'] == 1


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


@pytest.mark.parametrize('strategy', STRATEGIES)
@pytest.mark.parametrize('screened', [False, True])
def test_product_scratch_cap_tiles_columns_and_bounds_omissions(strategy, screened):
    ham = random_hamiltonian(5, seed=283)
    w = scsf.csf_wfn(5, 4, 0); w.add_all_csfs()
    ld = np.dtype(np.longdouble).itemsize
    one_column = len(w) * (8 + (ld if screened else 0)) + (4 * ld if screened else 0)
    op = scsf.csf_direct_op(ham, w, strategy=strategy, nthreads=3,
        product_tol=.02 if screened else 0., max_product_scratch_bytes=one_column * 2)
    matrix = scsf.CSFReferenceBackend(ham).block(w, w, 0)
    x = np.random.default_rng(284).normal(size=(len(w), 5))
    y = op.matmat(x)
    if screened:
        assert np.all(np.max(np.abs(y - matrix @ x), axis=0) <= op.last_error_inf + 5e-12)
        assert np.all(np.linalg.norm(y - matrix @ x, axis=0) <= op.last_error_l2 + 5e-12)
    else:
        npt.assert_allclose(y, matrix @ x, atol=5e-12)
    stats = op.last_product_stats
    assert stats['rhs_tiles'] == 3
    for tile in stats['tile_product_stats']:
        actual = tile['output_scratch_bytes'] + tile['screening_error_scratch_bytes']
        assert actual == tile['forecast_product_scratch_bytes'] <= one_column * 2
    if not screened:
        assert stats['screening_error_scratch_bytes'] == 0
        assert stats['exact_product_kernel'] == 1
    # A subsequent narrower call still fits and recovers the full worker count
    # when its width allows it, without retaining the previous output peak.
    op.matvec(x[:, 0])
    assert op.last_product_stats['forecast_product_scratch_bytes'] <= one_column * 2
    with pytest.raises(MemoryError, match='one private output column'):
        scsf.csf_direct_op(ham, w, product_tol=0., max_product_scratch_bytes=1).matvec(x[:, 0])


@pytest.mark.parametrize('strategy', STRATEGIES[:3])
@pytest.mark.parametrize('fallback', [False, True])
@pytest.mark.parametrize('nelec,two_s', [(4, 0), (3, 1), (4, 2)])
def test_exact_gather_owns_rows_with_incomplete_spins(strategy, fallback, nelec, two_s):
    ham = random_hamiltonian(4, seed=285)
    full = scsf.csf_wfn(4, nelec, two_s); full.add_all_csfs()
    w = scsf.csf_wfn(4, nelec, two_s)
    for label in tuple(full)[::2]: w.add_csf(label)
    keys, hdet, _ = fock_reference(ham, nelec, two_s)
    u = transform_in_order(w, keys, two_s); matrix = u.T @ hdet @ u
    backend = scsf.CSFSpinBackend(ham, nthreads=3, max_index_bytes=0 if fallback else 2**20,
        max_reuse_bytes=0 if fallback else 2**16)
    op = scsf.csf_direct_op(ham, w, backend=backend, strategy=strategy, product_tol=0.,
        product_accumulation='gather', max_product_scratch_bytes=0,
        max_path_bytes=0 if fallback else 2**20, max_graph_bytes=0 if fallback else 2**20,
        batch_min_pairs=1, profile_couplings=True)
    x = np.random.default_rng(286).normal(size=(len(w), 4))
    assert not op.product_info['unique_occupation_blocks']
    for v in (x, x[:, :1], np.zeros_like(x), -x, np.eye(len(w))[:, :2]):
        y = op.matmat(v)
        npt.assert_allclose(y, matrix @ v, atol=5e-12)
        assert op.last_product_stats['output_scratch_bytes'] == 0
        assert op.last_product_stats['screening_error_scratch_bytes'] == 0
        assert op.last_product_stats['forecast_product_scratch_bytes'] == 0
    npt.assert_allclose(x[:, 0] @ op(x[:, 1]), op(x[:, 0]) @ x[:, 1], atol=5e-12)
    assert op.matmat(np.empty((len(w), 0))).shape == (len(w), 0)
    if strategy == 'compact-adaptive-guga':
        assert op.product_info['product_team_creations'] == 1


@pytest.mark.parametrize('encoder', ['portable', 'bmi2'])
@pytest.mark.parametrize('strategy', STRATEGIES[:3])
def test_encoder_override_preserves_memo_layout_and_phases(encoder, strategy):
    ham = random_hamiltonian(5, seed=287)
    w = scsf.csf_wfn(5, 4, 0); w.add_all_csfs()
    auto = scsf.csf_direct_op(ham, w, strategy=strategy, product_tol=0., nthreads=2, batch_min_pairs=1)
    if encoder == 'bmi2' and not auto.product_info['bmi2_available']:
        with pytest.raises(ValueError, match='unavailable'):
            scsf.csf_direct_op(ham, w, spin_encoder=encoder)
        return
    op = scsf.csf_direct_op(ham, w, strategy=strategy, product_tol=0., nthreads=2,
        spin_encoder=encoder, batch_min_pairs=1)
    x = np.random.default_rng(288).normal(size=(len(w), 3))
    for _ in range(2): npt.assert_allclose(op.matmat(x), auto.matmat(x), atol=3e-12)
    for key in ('spin_slot_bytes', 'memo_slots', 'aligned_tiny_cache', 'paired_spin_encoding'):
        assert op.product_info[key] == auto.product_info[key]
    assert op.product_info['bmi2_enabled'] == (encoder == 'bmi2')


def test_adaptive_affinity_and_team_survive_width_activity_and_exceptions():
    import os
    if not hasattr(os, 'sched_getaffinity'): pytest.skip('Linux affinity required')
    allowed = os.sched_getaffinity(0)
    cpus = sorted(allowed)[:3]
    ham = random_hamiltonian(6, seed=289)
    w = scsf.csf_wfn(6, 4, 0); w.add_all_csfs()
    matrix = scsf.CSFReferenceBackend(ham).block(w, w, 0)
    op = scsf.csf_direct_op(ham, w, strategy='compact-adaptive-guga', product_tol=0.,
        nthreads=len(cpus), worker_cpus=cpus, profile_couplings=True)
    dense = np.random.default_rng(290).normal(size=(len(w), 4))
    sparse = np.zeros((len(w), 1)); sparse[0] = 1.
    for x in (dense, sparse, dense[:, :1], sparse, dense):
        npt.assert_allclose(op.matmat(x), matrix @ x, atol=6e-12)
        assert os.sched_getaffinity(0) == allowed
        stats = op.last_product_stats
        for i, cpu in enumerate(cpus):
            if stats.get(f'worker_{i}_rows', 0):
                assert stats[f'worker_{i}_cpu_first'] == stats[f'worker_{i}_cpu_last'] == cpu
        assert stats['worker_cpu_migrations'] == 0
        assert op.product_info['product_team_creations'] == 1
        assert op.product_info['retained_product_team_workers'] == len(cpus)
        phases = [v for k, v in stats.items() if k.startswith('wall_') and k.endswith('_ns') and k != 'wall_native_ns']
        assert sum(phases) <= stats['wall_native_ns']
    backend = scsf.CSFSpinBackend(ham, max_bond_elements=1, nthreads=len(cpus))
    with pytest.raises(MemoryError):
        scsf.csf_direct_op(ham, w, backend=backend, strategy='compact-adaptive-guga',
            product_tol=0., worker_cpus=cpus)
    assert os.sched_getaffinity(0) == allowed
    npt.assert_allclose(op.matmat(dense), matrix @ dense, atol=6e-12)


@pytest.mark.parametrize('k', [63, 64, 65])
@pytest.mark.parametrize('accumulation', ['private', 'gather'])
@pytest.mark.parametrize('paths', [0, 2**20])
def test_orbital_word_boundaries_with_two_electron_closure(k, accumulation, paths):
    small = random_hamiltonian(4, seed=291)
    active = [0, 1, k-2, k-1]
    one, two = np.zeros((k, k)), np.zeros((k,)*4)
    one[np.ix_(active, active)] = small.one_mo
    two[np.ix_(active, active, active, active)] = small.two_mo
    ham = scsf.hamiltonian(0., one, two)
    short = scsf.csf_wfn(4, 4, 0)
    w = scsf.csf_wfn(k, 4, 0)
    for label in ('uudd', 'udud', '2200', '0022', 'u2d0'):
        short.add_csf(label)
        expanded = ['0'] * k
        for p, char in zip(active, label): expanded[p] = char
        w.add_csf(''.join(expanded))
    keys, hdet, _ = fock_reference(small, 4, 0)
    u = transform_in_order(short, keys, 0); matrix = u.T @ hdet @ u
    op = scsf.csf_direct_op(ham, w, strategy='compact-adaptive-guga', product_tol=0.,
        nthreads=3, product_accumulation=accumulation, max_path_bytes=paths,
        spin_storage='checkpoints', max_graph_bytes=0)
    x = np.random.default_rng(292).normal(size=(len(w), 3))
    npt.assert_allclose(op.matmat(x), matrix @ x, atol=5e-12)


def test_scratch_forecast_rejects_impossible_caps_and_invalid_controls():
    ham = random_hamiltonian(4)
    w = scsf.csf_wfn(4, 4, 0); w.add_all_csfs()
    for options in (dict(max_product_scratch_bytes=-1), dict(spin_encoder='unknown'),
                    dict(product_accumulation='unknown'), dict(product_accumulation='gather',strategy='tuga'),
                    dict(worker_cpus=[0,0],nthreads=2,strategy='compact-adaptive-guga')):
        with pytest.raises(ValueError): scsf.csf_direct_op(ham, w, **options)
    op = scsf.csf_direct_op(ham, w, product_tol=0., nthreads=3, max_product_scratch_bytes=8*len(w)*2)
    forecast = op.product_memory_forecast(8)
    assert forecast['rhs_tile_width'] == 2 and forecast['workers'] == 1 and forecast['fits']
    assert forecast['forecast_product_scratch_bytes'] == 16*len(w)
    x = np.random.default_rng(293).normal(size=(len(w), 8))
    op.matmat(x)
    assert op.last_product_stats['forecast_product_scratch_bytes'] == forecast['forecast_product_scratch_bytes']
    assert not scsf.csf_direct_op(ham,w,max_product_scratch_bytes=0).product_memory_forecast()['fits']
    screened = scsf.csf_direct_op(ham,w,product_accumulation='gather')
    assert not screened.product_memory_forecast()['fits']
    with pytest.raises(ValueError,match='exact product'): screened.matvec(x[:,0])
    # Unscreening is an exact gather call even if defaults use screening.
    npt.assert_allclose(screened.matmat(x,unscreened=True),op.matmat(x),atol=5e-12)


@pytest.mark.parametrize('strategy', STRATEGIES[:3])
@pytest.mark.parametrize('storage,encoder,cap,collect', [
    ('full', 'portable', 0, True), ('full', 'auto', 16, False),
    ('checkpoints', 'portable', 2**16, False), ('checkpoints', 'auto', 2**16, True)])
def test_short_generator_kernel_matches_oracle_and_general(strategy, storage, encoder, cap, collect):
    ham = random_hamiltonian(5, seed=294)
    w = scsf.csf_wfn(5, 4, 0); w.add_all_csfs()
    keys, hdet, _ = fock_reference(ham, 4, 0)
    u = transform_in_order(w, keys, 0); matrix = u.T @ hdet @ u
    backend = scsf.CSFSpinBackend(ham, max_reuse_bytes=cap, nthreads=2)
    settings = dict(backend=backend, strategy=strategy, product_tol=0.,
        batch_min_pairs=100000, spin_storage=storage, spin_encoder=encoder, collect_spin_stats=collect)
    op = scsf.csf_direct_op(ham, w, compact_spin_kernel='short', **settings)
    control = scsf.csf_direct_op(ham, w, **settings)
    x = np.random.default_rng(295).normal(size=(len(w), 4))
    for _ in range(2):
        npt.assert_allclose(op.matmat(x), matrix @ x, atol=5e-12)
        npt.assert_allclose(op.matmat(x), control.matmat(x), atol=5e-12)
        assert op.last_product_stats['short_spin_entries'] > 0
    assert op.product_info['memo_bytes'] == control.product_info['memo_bytes']
