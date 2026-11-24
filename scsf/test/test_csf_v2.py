"""Signed native occupation-block, growth and CIH cross-checks."""
from itertools import combinations, islice, product
from pathlib import Path

import numpy as np
import pytest

import scsf
from .reference.oracle import fock_operators, hamiltonian_block, random_hamiltonian


def wavefunction(k, n, s, labels):
    w = scsf.csf_wfn(k, n, s)
    for label in labels:
        w.add_csf(label)
    return w


@pytest.mark.parametrize('k,n,s', [(4,4,0), (4,3,1), (5,4,2), (3,0,0), (0,0,0)])
@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('kernel', scsf.csf_structured_op.kernels)
def test_native_layout_against_fock_space(k, n, s, partial, kernel):
    ham = random_hamiltonian(k, seed=1234+n)
    full = scsf.csf_wfn(k, n, s)
    full.add_all_csfs()
    labels = tuple(full)[::-2] if partial else tuple(full)
    w = wavefunction(k, n, s, labels)
    ref = hamiltonian_block(ham, labels, labels, s)
    op = scsf.csf_structured_op(ham, w, strategy=kernel, nthreads=2)
    assert tuple(op.labels()) == labels
    if not partial:
        assert op.product_info['rank_map_bytes'] == 0
        assert op.product_info['public_permutation_bytes'] == 0
    order = [w.index_csf(label) for label in op.labels(native=True)]
    native_ref = ref[np.ix_(order, order)]
    np.testing.assert_allclose(op.diagonal(), np.diag(ref), atol=1e-12)
    for width in (0, 1, 2, 8):
        x = np.random.default_rng(32+width).normal(size=(len(w), width))
        np.testing.assert_allclose(op.matmat(x), ref@x, atol=1e-12)
        np.testing.assert_allclose(op.native_matmat(x[order]), native_ref@x[order], atol=1e-12)


def test_wide_occupations_and_empty_space():
    k = 66
    one = np.diag(np.arange(k, dtype=float))
    pairs = k * (k + 1) // 2
    ham = scsf.PackedHamiltonian(0., one, np.zeros(pairs * (pairs + 1) // 2))
    w = scsf.csf_wfn(k, 2, 0)
    op = scsf.csf_structured_op(ham, w)
    assert op.matmat(np.empty((0, 1))).shape == (0, 1)
    assert op.diagonal().size == 0
    w.add_csf('0'*65+'2')
    w.add_csf('u'+'0'*64+'d')
    op.update(ham, w)
    assert op.product_info['wide_mask_bytes'] > 0
    np.testing.assert_allclose(op.matvec([1., -1.]), [130., -65.])
    np.testing.assert_allclose(op.diagonal(), [130., 65.])


@pytest.mark.parametrize('complete', [False, True])
def test_native_growth_strict_hci_and_remap(complete):
    ham = random_hamiltonian(4, seed=197)
    seed = ('2200',) if complete else ('udud', '2200')
    w = wavefunction(4, 4, 0, seed)
    full = scsf.csf_wfn(4, 4, 0)
    full.add_all_csfs()
    external = [label for label in full if label not in seed]
    elements = hamiltonian_block(ham, external, seed, 0)
    weights = np.linspace(1., .7, len(seed))
    found = {label for label, row in zip(external, elements)
             if np.any(np.abs(row)*weights > .04)}
    if complete:
        occupied = {scsf.csf._configuration(label) for label in found}
        found = {label for label in external if scsf.csf._configuration(label) in occupied}
    assert found
    op = scsf.csf_structured_op(ham, w, nthreads=2)
    old_diag = op.native_diagonal()
    remap, stats = op.select_hci(weights, .04, selection_unit='configuration' if complete else 'csf')
    assert stats['added'] == len(found)
    labels = tuple(op.labels())
    assert labels[:len(seed)] == seed
    assert set(labels[len(seed):]) == found
    if complete:
        assert remap.size == 0
        assert op.product_info['rank_map_bytes'] == op.product_info['public_permutation_bytes'] == 0
    else:
        assert tuple(op.labels(native=True))[0] == 'uudd'
        assert remap.tolist() == [1, 2]
    old_rows = remap if remap.size else np.arange(len(seed))
    np.testing.assert_array_equal(op.native_diagonal()[old_rows], old_diag)
    fresh = scsf.csf_structured_op(ham, wavefunction(4, 4, 0, labels))
    x = np.random.default_rng(12).normal(size=(len(labels), 2))
    np.testing.assert_allclose(op.matmat(x), fresh.matmat(x), atol=1e-12)
    np.testing.assert_allclose(op.diagonal(), fresh.diagonal(), atol=1e-12)


@pytest.mark.parametrize('k,hole,particle', [(4, 0, 1), (65, 64, 0)])
def test_hci_single_transfer_threshold_is_strict(k, hole, particle):
    one = np.diag(np.arange(k, dtype=float))
    one[hole, particle] = one[particle, hole] = .125
    ham = scsf.Hamiltonian(0., one, np.zeros((k,)*4))
    initial = '0'*hole + 'u' + '0'*(k-hole-1)
    target = '0'*particle + 'u' + '0'*(k-particle-1)
    basis = wavefunction(k, 1, 1, [initial])
    op = scsf.csf_structured_op(ham, basis, nthreads=1)
    for eps in (np.nextafter(.125, np.inf), .125):
        _, stats = op.select_hci([1.], eps)
        assert stats['added'] == 0
        assert stats['ordered_singles']
    _, stats = op.select_hci([1.], np.nextafter(.125, 0.))
    assert stats['added'] == 1
    assert tuple(op.labels(native=True)) == (initial, target)


def test_nminus1_reuses_integral_layout_after_growth():
    ham = random_hamiltonian(4, seed=197)
    basis = wavefunction(4, 4, 0, ['2200'])
    op = scsf.csf_structured_op(ham, basis, strategy='nminus1', nthreads=1)
    old = op.product_info
    _, stats = op.select_hci([1.], .04, selection_unit='configuration')
    assert stats['added'] > 0
    new = op.product_info
    assert new['reused_auxiliary_integrals'] == 1
    assert new['auxiliary_integral_bytes'] == old['auxiliary_integral_bytes'] > 0
    assert new['removal_metadata_overlap_bytes'] == (
        old['metadata_bytes'] + new['metadata_build_bytes'] - new['auxiliary_integral_bytes'])
    labels = list(op.labels(native=True))
    fresh = scsf.csf_structured_op(ham, wavefunction(4, 4, 0, labels), strategy='nminus1', nthreads=1)
    trial = np.random.default_rng(5).normal(size=(len(labels), 1))
    np.testing.assert_allclose(op.native_matmat(trial), fresh.native_matmat(trial), atol=1e-12)


@pytest.mark.parametrize('kernel', scsf.csf_structured_op.kernels)
def test_signed_native_cih_fixture(kernel):
    root = Path(__file__).resolve().parents[2]
    data = (root/'benchmarks/records/cih-lih-signed-2026-11-19.txt').read_text().splitlines()
    k, s, ncsf = map(int, data[0].split())
    rows = [row.split() for row in data[1:ncsf+1]]
    labels = []
    for first, second, *_ in rows:
        a, b = int(first), int(second)
        labels.append(''.join('0ud2'[((a>>p)&1)+2*((b>>p)&1)] for p in range(k)))
    d, x, y = np.split(np.asarray([row[2:] for row in rows], float), [1, 3], axis=1)
    with np.load(root/'scsf/test/data/lih_sto3g.npz') as snapshot:
        ham = scsf.hamiltonian(snapshot['ecore'], snapshot['one'], snapshot['two'])
        w = wavefunction(k, int(snapshot['nelec']), s, labels)
    op = scsf.csf_structured_op(ham, w, strategy=kernel, nthreads=1)
    assert op.product_info['complete_configurations'] == op.product_info['configurations'] == 38
    assert op.product_info['rank_map_bytes'] == 0
    assert op.product_info['public_permutation_bytes'] > 0
    p, q, bra, ket, value = data[ncsf+1].split()
    assert (labels[int(bra)], labels[int(ket)]) == ('u2d000', '2ud000')
    assert float(value) == pytest.approx(-1.)
    generators = {}
    for line in data[ncsf+1:]:
        p, q, bra, ket, value = line.split()
        pair = int(p), int(q)
        if pair not in generators:
            one = np.zeros((k, k))
            one[pair] = 1.
            generators[pair] = scsf.csf_one_electron_product(one, w, np.eye(ncsf))
        assert generators[pair][int(bra), int(ket)] == pytest.approx(float(value), abs=1e-13)
    ref = hamiltonian_block(ham, labels, labels, s)
    np.testing.assert_allclose(np.diag(ref), d[:,0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(ref@x, y, atol=1e-12, rtol=0)
    np.testing.assert_allclose(op.diagonal(), d[:,0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(op.matmat(x), y, atol=1e-12, rtol=0)


def test_native_builder_failure_preserves_generation():
    ham = random_hamiltonian(4, seed=197)
    w = wavefunction(4, 4, 0, ('2200',))
    probe = scsf.csf_structured_op(ham, w, nthreads=1)
    probe.select_hci([1.], 1e8)
    cap = max(probe.product_info['peak_tracked_memory_bytes'],
              probe.product_info['tracked_memory_bytes'])
    op = scsf.csf_structured_op(ham, w, max_mem=cap, nthreads=1)
    expected = op.matvec([1.])
    with pytest.raises(MemoryError, match='max-mem'):
        op.select_hci([1.], .04, selection_unit='configuration')
    assert tuple(op.labels()) == ('2200',)
    np.testing.assert_array_equal(op.matvec([1.]), expected)


@pytest.mark.parametrize('kernel', ['cih', 'particle-hole'])
def test_neighbor_and_plan_caches_share_the_memory_cap(kernel):
    w = wavefunction(4, 4, 0, ['udud','2200','0022','uudd','2002','u0d2'])
    ham = random_hamiltonian(4, seed=653)
    x = np.random.default_rng(46).normal(size=(len(w), 8))
    ref = hamiltonian_block(ham, tuple(w), tuple(w), 0)@x
    probe = scsf.csf_structured_op(ham, w, strategy=kernel, nthreads=3)
    probe.matmat(x)
    cap = probe.product_info['peak_tracked_memory_bytes']
    op = scsf.csf_structured_op(ham, w, strategy=kernel, nthreads=3, max_mem=cap)
    for _ in range(2):
        np.testing.assert_allclose(op.matmat(x), ref, atol=1e-12)
        assert op.product_info['peak_tracked_memory_bytes'] == cap
    if kernel == 'cih':
        assert op.last_product_stats['plan_hits'] == op.product_info['configurations']
    else:
        assert op.product_info['cached_strings'] == op.product_info['main_strings']
    with pytest.raises(MemoryError):
        insufficient = scsf.csf_structured_op(ham, w, strategy=kernel, nthreads=3, max_mem=cap-1)
        insufficient.matmat(x)


def test_hci_tables_share_the_memory_cap_with_operator():
    ham = random_hamiltonian(5)
    w = wavefunction(5, 4, 0, ['22000'])
    probe = scsf.csf_structured_op(ham, w, strategy='cih', nthreads=1)
    cap = probe.product_info['peak_tracked_memory_bytes']
    limited = scsf.csf_structured_op(ham, w, strategy='cih', nthreads=1, max_mem=cap)
    with pytest.raises(MemoryError, match='HCI'):
        limited.select_hci([1.], .1)
    assert tuple(limited.labels()) == ('22000',)


def test_cih_reuses_vector_workspace_and_releases_it_for_other_operations():
    ham = random_hamiltonian(5)
    w = wavefunction(5, 4, 0, ['22000', '20200', '20020'])
    op = scsf.csf_structured_op(ham, w, strategy='cih', nthreads=2)
    matrix = hamiltonian_block(ham, tuple(w), tuple(w), 0)
    vector = np.arange(len(w), dtype=float)
    for trial in (vector, -vector):
        np.testing.assert_allclose(op.matvec(trial), matrix@trial, atol=1e-12)
        info = op.product_info
        assert info['retained_workspace_bytes'] == 2*len(w)*8
        assert info['tracked_memory_bytes'] <= info['peak_tracked_memory_bytes']
    block = np.column_stack((vector, -vector))
    np.testing.assert_allclose(op.matmat(block), matrix@block, atol=1e-12)
    assert op.product_info['retained_workspace_bytes'] == 0
    op.matvec(vector)
    op.select_hci(np.zeros(len(w)), .1)
    assert op.product_info['retained_workspace_bytes'] == 0


def test_nminus2_private_vector_workspace_survives_retry_and_releases():
    ham = random_hamiltonian(5)
    w = scsf.csf_wfn(5, 4, 0)
    w.add_all_csfs()
    opts = dict(strategy='nminus2', pair_accumulation='private', nthreads=3)
    probe = scsf.csf_structured_op(ham, w, **opts)
    x = np.random.default_rng(918).normal(size=len(w))
    reference = probe.matvec(x)
    expected_bytes = len(w)*3*8
    assert probe.product_info['retained_workspace_bytes'] == expected_bytes
    cap = probe.product_info['peak_tracked_memory_bytes']
    limited = scsf.csf_structured_op(ham, w, max_mem=cap, **opts)
    np.testing.assert_allclose(limited.matvec(x), reference, atol=1e-12)
    columns = cap//expected_bytes+1
    with pytest.raises(MemoryError, match='product arrays'):
        limited.matmat(np.zeros((len(w), columns)))
    np.testing.assert_allclose(limited.matvec(-x), -reference, atol=1e-12)
    assert limited.product_info['retained_workspace_bytes'] == expected_bytes
    assert limited.matmat(np.empty((len(w), 0))).shape == (len(w), 0)
    assert limited.product_info['retained_workspace_bytes'] == 0
    block = np.column_stack((x, -x))
    np.testing.assert_allclose(probe.matmat(block), np.column_stack((reference, -reference)), atol=1e-12)
    assert probe.product_info['retained_workspace_bytes'] == 0
    probe.matvec(x)
    probe.select_hci(np.zeros(len(w)), .1, selection_unit='configuration')
    assert probe.product_info['retained_workspace_bytes'] == 0
    serial = scsf.csf_structured_op(ham, w, strategy='nminus2', nthreads=1)
    np.testing.assert_allclose(serial.matvec(x), reference, atol=1e-12)
    assert serial.product_info['retained_workspace_bytes'] == 0


def test_grouped_cih_reuses_only_plans_unaffected_by_appended_occupations():
    one = np.zeros((6, 6))
    one[0, 3] = one[3, 0] = .5
    ham = scsf.Hamiltonian(0., one, np.zeros((6,)*4))
    w = wavefunction(6, 6, 0, ['222000', '000222', 'ud2002'])
    op = scsf.csf_structured_op(ham, w, strategy='cih', nthreads=2)
    op.matvec([1., 0., 0.])
    _, stats = op.select_hci([1., 0., 0.], .1, selection_unit='configuration')
    assert stats['added'] == 1
    assert op.product_info['reused_plans'] == 1
    assert op.product_info['reused_spin_bank'] == 1
    labels = tuple(op.labels())
    ref = hamiltonian_block(ham, labels, labels, 0)
    np.testing.assert_allclose(op.matmat(np.eye(len(labels))), ref, atol=1e-12)
    assert op.last_product_stats['plans_built'] == len(labels)-1


def test_grouped_cih_suffix_index_reuses_only_unaffected_plans():
    k = 19
    ham = random_hamiltonian(k, seed=217)
    w = scsf.csf_wfn(k, 4, 0)

    def add(sites):
        w.add_configuration(tuple(int(i in sites) for i in range(k)))

    add((0, 1, 2, 3))
    add((4, 5, 6, 7))
    op = scsf.csf_structured_op(ham, w, strategy='cih', nthreads=2)
    op.matvec(np.ones(len(w)))

    for sites in islice(combinations(range(4, k), 4), 1, 1026):
        add(sites)
    op.update(ham, w)
    assert op.product_info['reused_spin_bank'] == 1
    assert op.product_info['reused_plans'] == 1
    ref = scsf.csf_structured_op(ham, w, strategy='nminus2', nthreads=2)
    x = np.random.default_rng(926).normal(size=(len(w), 2)) / np.sqrt(len(w))
    np.testing.assert_allclose(op.matmat(x), ref.matmat(x), atol=1e-11, rtol=1e-11)
    assert op.last_product_stats['plans_built'] == 1026


def test_grouped_cih_width_one_complete_eight_open_shells():
    k = 9
    ham = random_hamiltonian(k, seed=256)
    w = scsf.csf_wfn(k, 8, 0)
    for vacancy in (0, 2, 5, 8):
        w.add_configuration(tuple(int(i != vacancy) for i in range(k)))
    assert len(w) == 56
    opts = dict(nthreads=3)
    cih = scsf.csf_structured_op(ham, w, strategy='cih', **opts)
    nminus2 = scsf.csf_structured_op(ham, w, strategy='nminus2', **opts)
    vector = np.random.default_rng(256).normal(size=len(w))
    coordinate = np.zeros(len(w))
    coordinate[19] = -1.
    for trial in (vector, coordinate):
        np.testing.assert_allclose(cih.matvec(trial), nminus2.matvec(trial),
                                   atol=1e-11, rtol=1e-11)


def test_wide_grouped_cih():
    k = 66
    active = [0, 3, 63, 65]
    small = random_hamiltonian(len(active), seed=209)
    wsmall = scsf.csf_wfn(len(active), 4, 0)
    wsmall.add_all_csfs()
    labels = tuple(wsmall)[::-2]
    wwide = scsf.csf_wfn(k, 4, 0)
    for label in labels:
        wide = ['0']*k
        for p, s in zip(active, label):
            wide[p] = s
        wwide.add_csf(wide)
    one = np.zeros((k,k))
    one[np.ix_(active, active)] = small.one_mo
    def pair(i, j):
        p, q = max(i,j), min(i,j)
        return p*(p+1)//2+q
    n = pair(k-1,k-1)+1
    packed = np.zeros(n*(n+1)//2)
    for p, pi in enumerate(active):
        for q, qi in enumerate(active):
            for r, ri in enumerate(active):
                for s, si in enumerate(active):
                    packed[pair(pair(pi,qi),pair(ri,si))] = small.two_mo[p,r,q,s]
    wide_ham = scsf.PackedHamiltonian(small.ecore, one, packed)
    op = scsf.csf_structured_op(wide_ham, wwide, strategy='cih', nthreads=2)
    x = np.random.default_rng(36).normal(size=(len(labels), 2))
    expected = hamiltonian_block(small, labels, labels, 0)
    np.testing.assert_allclose(op.matmat(x), expected@x, atol=1e-12)
    np.testing.assert_allclose(op.diagonal(), np.diag(expected), atol=1e-12)
    assert op.product_info['wide_grouped_uga']
    assert op.last_product_stats['generator_groups'] > 0


def test_nminus2_owner_gather_requires_space_for_full_trial_block():
    h = random_hamiltonian(5, seed=19)
    w = scsf.csf_wfn(5, 4, 0)
    w.add_all_csfs()
    probe = scsf.csf_structured_op(h, w, pair_accumulation='gather', nthreads=2)
    info = probe.product_info
    cap = max(info['peak_tracked_memory_bytes'],
              info['tracked_memory_bytes'] + info['product_array_bytes_per_rhs'])
    op = scsf.csf_structured_op(h, w, pair_accumulation='gather', nthreads=2, max_mem=cap)
    columns = (cap-info['tracked_memory_bytes'])//info['product_array_bytes_per_rhs'] + 1
    with pytest.raises(MemoryError, match='product arrays'):
        op.matmat(np.ones((len(w), columns)))
    x = np.random.default_rng(19).normal(size=len(w))
    ref = hamiltonian_block(h, tuple(w), tuple(w), 0)@x
    np.testing.assert_allclose(op.matvec(x), ref, atol=1e-12)


def test_nminus2_rank_only_replacement_shares_immutable_topology():
    one, two = np.zeros((4,4)), np.zeros((4,)*4)
    # Interchange spins on sites 1 and 2 within the singly occupied block.
    for p, q in ((1, 2), (2, 1)):
        for r, s in ((1, 2), (2, 1)):
            two[p,r,q,s] = .5
    ham = scsf.Hamiltonian(0., one, two)
    w = wavefunction(4, 4, 0, ['udud'])
    op = scsf.csf_structured_op(ham, w)
    _, stats = op.select_hci([1.], .01)
    assert stats['added'] == 1
    assert op.product_info['reused_removal_topology']
    assert op.product_info['metadata_build_bytes'] == 0
    ref = hamiltonian_block(ham, tuple(op.labels()), tuple(op.labels()), 0)
    np.testing.assert_allclose(op.matmat(np.eye(2)), ref, atol=1e-13)


@pytest.mark.parametrize('kernel', scsf.csf_structured_op.kernels)
def test_packed_integrals_all_kernels(kernel):
    k = 4
    tensor = np.random.default_rng(57).normal(size=(k, 2))
    core = np.array([[1., .4], [.4, .8]])
    one = np.diag([-.7,-.3,.2,.5])
    g = np.einsum('pu,qu,uv,rv,sv->prqs', tensor, tensor, core, tensor, tensor)
    exact = scsf.Hamiltonian(.1, one, g)
    w = wavefunction(k, 4, 0, ['udud', '2200', 'ud02', '2u0d'])
    x = np.random.default_rng(56).normal(size=(len(w), 4))
    ref = hamiltonian_block(exact, tuple(w), tuple(w), 0)
    ham = scsf.PackedHamiltonian.from_dense(exact)
    op = scsf.csf_structured_op(ham, w, strategy=kernel, nthreads=2)
    np.testing.assert_allclose(op.matmat(x), ref@x, atol=2e-12)
    np.testing.assert_allclose(op.diagonal(), np.diag(ref), atol=2e-12)
    if kernel == 'cih':
        assert op.product_info['grouped_dense_eri_bytes'] == 0


@pytest.mark.parametrize('kernel', ['cih', 'particle-hole', 'nminus1', 'nminus2'])
def test_operator_rejects_insufficient_memory(kernel):
    h = random_hamiltonian(4)
    w = wavefunction(4, 4, 0, ['2200', 'udud', '0022', 'ud20'])
    with pytest.raises(MemoryError):
        scsf.csf_structured_op(h, w, strategy=kernel, max_mem=0)


def test_exact_block_preconditioner():
    from scsf.drivers.solvers import solve_ci
    rng = np.random.default_rng(17)
    a = rng.normal(size=(30, 30))*.05
    a = a+a.T+np.diag(np.linspace(-2.,5.,30))
    rows = np.array([0, 2, 7, 9])
    block = a[np.ix_(rows, rows)]
    energies, vectors, converged = solve_ci(a.__matmul__, np.diag(a), 3, tol=1e-10,
                                          preconditioner_block=(rows, block))
    assert converged
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(a)[:3], atol=1e-12)
    assert np.max(np.linalg.norm(a@vectors-vectors*energies, axis=0)) < 1e-10


@pytest.mark.parametrize('spin,labels', [
    (0, ['udud', '0220', 'ud02', '2u0d']),
    (2, ['uu20', '2uu0', '0uu2']),
])
def test_configuration_batched_rdms_energy_and_trace(spin, labels):
    h = random_hamiltonian(4, seed=47)
    w = wavefunction(4, 4, spin, labels)
    op = scsf.csf_structured_op(h, w, strategy='cih')
    x = np.random.default_rng(561).normal(size=len(w))
    x /= np.linalg.norm(x)
    d1, d2, stats = op.native_rdms(x)
    np.testing.assert_allclose(op.native_rdm1(x), d1, atol=1e-13)
    np.testing.assert_allclose(d1, d1.T, atol=1e-13)
    np.testing.assert_allclose(d2, d2.transpose(2, 3, 0, 1), atol=1e-13)
    np.testing.assert_allclose(np.einsum('prqr->pq',d2), 3*d1, atol=1e-12)
    assert np.trace(d1) == pytest.approx(4.)
    assert stats['intermediate_occupations'] > len(w._configs)
    h_rdm = np.sum(d1*h.one_mo)+.5*np.sum(d2*h.two_mo)
    expected = x@op.native_matmat(x[:,None])[:,0]
    assert h_rdm == pytest.approx(expected, abs=1e-12)
    with pytest.raises(MemoryError):
        op.native_rdms(x, max_output_bytes=8)


@pytest.mark.parametrize('nthreads', [1, 4])
def test_transition_rdms_against_determinants(nthreads):
    k, n, spin = 4, 4, 0
    h = random_hamiltonian(k, seed=67)
    # Include a partial spin block and disjoint bra/ket support.
    w = wavefunction(k, n, spin, ['2200', 'udud', 'ud20', '2u0d', '0022', 'u0d2'])
    op = scsf.csf_structured_op(h, w, strategy='cih', nthreads=nthreads)
    bra = np.array([.3, -.2, .7, 0., 0., 0.])
    ket = np.array([0., 0., 0., -.5, .6, .4])
    x, y = map(op.to_native_order, (bra, ket))
    matrix = fock_operators(w)

    one = op.native_transition_rdm1(x, y)
    two = op.native_transition_rdm2(x, y, max_output_bytes=k**4*8)
    expected_one = np.empty_like(one)
    expected_two = np.empty_like(two)
    for p, q in product(range(k), repeat=2):
        expected_one[p, q] = bra @ matrix((p, q)) @ ket
    for p, r, q, s in product(range(k), repeat=4):
        expected_two[p, r, q, s] = bra @ matrix((p, q, r, s)) @ ket
    np.testing.assert_allclose(one, expected_one, atol=1e-13)
    np.testing.assert_allclose(two, expected_two, atol=1e-13)
    np.testing.assert_allclose(op.native_transition_rdm1(y, x), one.T, atol=1e-13)
    np.testing.assert_allclose(op.native_transition_rdm2(y, x), two.transpose(2, 3, 0, 1), atol=1e-13)
    np.testing.assert_allclose(np.einsum('prqr->pq', two), (n-1)*one, atol=1e-13)

    one, two, _ = op.native_rdms(x)
    np.testing.assert_allclose(op.native_transition_rdm1(x.copy(), x), one, atol=1e-13)
    np.testing.assert_allclose(op.native_rdm2(x), two, atol=1e-13)
    np.testing.assert_allclose(op.native_transition_rdm2(x.copy(), x), two, atol=1e-13)
    with pytest.raises(MemoryError, match='output budget'):
        op.native_rdm2(x, max_output_bytes=k**4*8-1)
    with pytest.raises(MemoryError, match='workspace budget'):
        op.native_transition_rdm2(x, y, max_workspace_bytes=0)
    with pytest.raises(ValueError, match='native-row'):
        op.native_transition_rdm1(x, y[:-1])
    with pytest.raises(TypeError, match='real'):
        op.native_transition_rdm2(x, y.astype(complex))


def test_parallel_rdm1_with_wide_occupations():
    k = 130
    pairs = k * (k + 1) // 2
    h = scsf.PackedHamiltonian(0., np.zeros((k, k)), np.zeros(pairs * (pairs + 1) // 2))
    w = scsf.csf_wfn(k, 1, 1)
    w.add_all_csfs()
    op = scsf.csf_structured_op(h, w, strategy='nminus2', nthreads=4)
    orbitals = [label.index('u') for label in op.labels(native=True)]
    bra = np.linspace(-.5, 1., k)
    ket = np.linspace(1.5, -.6, k)**2
    expected = np.outer(bra, ket)
    actual = op.native_transition_rdm1(bra[orbitals], ket[orbitals])
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    actual = op.native_rdm1(ket[orbitals])
    np.testing.assert_allclose(actual, np.outer(ket, ket), atol=1e-13)


@pytest.mark.parametrize('accumulation', ['private', 'gather'])
def test_new_occupations_merge_nminus2_removal_postings(accumulation):
    h = random_hamiltonian(5, seed=20)
    w = wavefunction(5, 4, 0, ['22000','20200','20020','20002','02200','02020'])
    op = scsf.csf_structured_op(h, w, pair_accumulation=accumulation, nthreads=2)
    full = scsf.csf_wfn(5, 4, 0)
    full.add_all_csfs()
    labels = list(full)
    rows = {label: i for i, label in enumerate(labels)}
    matrix = hamiltonian_block(h, labels, labels, 0)
    for _ in range(2):
        count = op.shape[0]
        internal = [rows[label] for label in op.labels(native=True)]
        external = sorted(set(range(len(full))) - set(internal))
        weights = np.linspace(.9, .1, count)
        scores = np.max(np.abs(matrix[np.ix_(external, internal)]) * weights, axis=1)
        threshold = np.sort(scores)[-2:].mean()
        _, stats = op.select_hci(weights, threshold)
        assert stats['added'] == 1
        assert list(op.labels())[-1] == labels[external[np.argmax(scores)]]
        assert op.product_info['incremental_removal_groups'] == 1
        assert op.product_info['removal_metadata_overlap_bytes'] > op.product_info['metadata_bytes']
        fresh_wfn = wavefunction(5, 4, 0, list(op.labels()))
        fresh = scsf.csf_structured_op(h, fresh_wfn, pair_accumulation=accumulation)
        trial = np.random.default_rng(19).normal(size=(count+1, 3))
        np.testing.assert_allclose(op.matmat(trial), fresh.matmat(trial), atol=1e-12)
        np.testing.assert_allclose(op.matvec(trial[:, 0]), fresh.matmat(trial)[:, 0], atol=1e-12)
        np.testing.assert_allclose(op.diagonal(), fresh.diagonal(), atol=1e-12)


def test_parallel_private_nminus2_postings_match_serial_and_append():
    k = 23
    rng = np.random.default_rng(601)
    patterns = set()
    while len(patterns) < 2400:
        patterns.add(tuple(sorted(rng.choice(k, size=8, replace=False))))
    labels = []
    for sites in sorted(patterns):
        label = ['0'] * k
        for site, step in zip(sites, 'uududdud'):
            label[site] = step
        labels.append(''.join(label))

    h = random_hamiltonian(k, seed=595)
    w = wavefunction(k, 8, 0, labels)
    op = scsf.csf_structured_op(h, w, strategy='nminus2', nthreads=3)
    assert op.product_info['parallel_posting_tasks'] >= 2

    def compare_with_serial():
        serial = scsf.csf_structured_op(h, w, strategy='nminus2', nthreads=1)
        x = rng.normal(size=(len(w), 2))
        x /= np.linalg.norm(x, axis=0)
        np.testing.assert_allclose(op.diagonal(), serial.diagonal(), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(op.matvec(x[:, 0]), serial.matvec(x[:, 0]), atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(op.matmat(x), serial.matmat(x), atol=1e-11, rtol=1e-11)

    compare_with_serial()
    w.add_csf('2222' + '0' * (k-4))
    op.update(h, w)
    assert op.product_info['incremental_removal_groups'] == 1
    compare_with_serial()


def test_nminus2_reuses_folded_groups_when_singletons_gain_parents():
    h = random_hamiltonian(8, seed=20)
    w = wavefunction(8, 4, 0, ['22000000', '00220000', '00002200', '00000022'])
    op = scsf.csf_structured_op(h, w, nthreads=3)
    assert op.product_info['folded_pair_terms'] > 0

    w.add_csf('20200000')
    op.update(h, w)
    info = op.product_info
    assert info['incremental_removal_groups'] == 1
    assert info['folded_reused_groups'] == 2
    fresh = scsf.csf_structured_op(h, w, nthreads=3)
    assert 0 < info['folded_staged_terms'] < fresh.product_info['folded_staged_terms']
    x = np.random.default_rng(19).normal(size=(len(w), 3))
    for width in (1, 2, 3):
        np.testing.assert_allclose(op.matmat(x[:, :width]), fresh.matmat(x[:, :width]), atol=1e-12)


@pytest.mark.parametrize('k', [32, 33, 64, 65, 138])
def test_nminus2_removal_classes_at_occupation_boundaries(k):
    rng = np.random.default_rng(104)
    sites = [0, k//2, k-2, k-1]
    labels = ['2200', 'udud', 'uudd', 'u02d', '0022']
    one = rng.normal(size=(4, 4))
    one = (one + one.T) / 2
    factors = rng.normal(size=(3, 10)) * .1
    pair_matrix = factors.T @ factors
    row, col = np.tril_indices(pair_matrix.shape[0])
    small_h = scsf.PackedHamiltonian(0., one, pair_matrix[row, col])
    small_w = wavefunction(4, 4, 0, labels)
    reference = scsf.csf_structured_op(small_h, small_w, strategy='cih')

    large_one = np.zeros((k, k))
    large_one[np.ix_(sites, sites)] = one
    pairs = k * (k + 1) // 2
    large_packed = np.zeros(pairs * (pairs + 1) // 2)
    p, q = np.take(sites, np.tril_indices(4))
    pair_ids = p * (p + 1) // 2 + q
    packed_rows = pair_ids[row]
    large_packed[packed_rows * (packed_rows + 1) // 2 + pair_ids[col]] = pair_matrix[row, col]
    large_h = scsf.PackedHamiltonian(0., large_one, large_packed)
    wide_labels = []
    for compact in labels:
        label = ['0'] * k
        for site, step in zip(sites, compact):
            label[site] = step
        wide_labels.append(''.join(label))
    w = wavefunction(k, 4, 0, wide_labels)
    op = scsf.csf_structured_op(large_h, w, strategy='nminus2', nthreads=2)
    x = rng.normal(size=(len(w), 2))
    np.testing.assert_allclose(op.diagonal(), reference.diagonal(), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(op.matvec(x[:, 0]), reference.matvec(x[:, 0]), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(op.matmat(x), reference.matmat(x), atol=1e-11, rtol=1e-11)


def test_cih_slot_sweep_matches_immediate_probes():
    h = random_hamiltonian(5, seed=15)
    w = wavefunction(5, 4, 0, ['22000','00220','uudd0','ud020','22000','u0d02'])
    a = scsf.csf_structured_op(h, w, strategy='cih')
    b = scsf.csf_structured_op(h, w, strategy='cih', cih_slot_sweep=True)
    x = np.random.default_rng(25).normal(size=(len(w), 2))
    np.testing.assert_allclose(b.matmat(x), a.matmat(x), atol=1e-13)
    assert b.last_product_stats['slot_sweep_probes'] > 0


def test_rdm_cache_uses_values_budgets_and_generation():
    h = random_hamiltonian(4, seed=36)
    w = wavefunction(4, 4, 0, ['2200','2020'])
    op = scsf.csf_structured_op(h, w, strategy='cih')
    x = np.array([.6,.8])
    a, b, stats = op.native_rdms(x, cache=True)
    assert stats['cache_hit'] == 0
    a.fill(7.)
    b.fill(2.)
    a, b, stats = op.native_rdms(x.copy(), cache=True)
    assert stats['cache_hit'] == 1
    assert np.trace(a) == pytest.approx(4.)
    with pytest.raises(MemoryError):
        op.native_rdms(x, max_output_bytes=16, cache=True)
    x[0] -= .2
    _, _, stats = op.native_rdms(x, cache=True)
    assert stats['cache_hit'] == 0
    w.add_csf('0220')
    op.update(h,w)
    x = np.r_[x, 0.]
    _, _, stats = op.native_rdms(x, cache=True)
    assert stats['cache_hit'] == 0


def test_jacobi_davidson_uses_exact_outer_residuals():
    from scsf.drivers.solvers import solve_ci
    rng = np.random.default_rng(37)
    h = rng.normal(size=(24,24))*.06
    h = h+h.T+np.diag(np.arange(24)*.21-2.)
    stats = {}
    e, c, ok = solve_ci(h.__matmul__, np.diag(h), 2, tol=1e-10, jd_iterations=3, stats=stats)
    assert ok and stats['jd_product_calls']>0
    np.testing.assert_allclose(e, np.linalg.eigvalsh(h)[:2], atol=1e-12)
    assert np.max(np.linalg.norm(h@c-c*e, axis=0))<1e-10


def test_damped_complex_response_davidson():
    from scsf.drivers.solvers import davidson_response
    h = np.array([[1.4,.8,-.2],[.8,2.7,.3],[-.2,.3,2.5]])-.4j*np.eye(3)
    rhs = np.array([.7+.08j, .4, -.9j])
    answer = davidson_response(h.__matmul__, rhs, np.diag(h), tol=1e-12)
    np.testing.assert_allclose(answer, np.linalg.solve(h, rhs), atol=1e-12)


def test_shared_generator_probe_matches_nminus2():
    k = 5
    factors = np.random.default_rng(23).normal(size=(2,k,k))
    factors += factors.swapaxes(1,2).copy()
    h = np.diag(np.linspace(-2.,2.,k))
    ham = scsf.Hamiltonian(0.,h,np.einsum('lpq,lrs->prqs',factors,factors))
    w = scsf.csf_wfn(k,4,0)
    w.add_all_csfs()
    op = scsf.csf_structured_op(ham, w)
    x = np.random.default_rng(8).normal(size=len(w))
    x[::3] = 0.
    actual, _ = op._state[1]._generator_probe(x, 8*1024**2, 8192, 1024**2)
    np.testing.assert_allclose(actual, op.native_matmat(x[:,None])[:,0], atol=6e-12)
    with pytest.raises(MemoryError):
        op._state[1]._generator_probe(x, 64, 8192, 1024**2)
