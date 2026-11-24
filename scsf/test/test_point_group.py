"""Exact point-group selection rules, ERI packing, and transition densities."""
import numpy as np
import pytest

import scsf
from scsf.drivers.csf_driver import natural_orbitals
from scsf.drivers.csf_driver import main as run_sci
from scsf.drivers.response import resp
from scsf.integrals import rotate_hamiltonian
from .reference.oracle import hamiltonian_block


def spatial_integrals(symmetry, seed=7):
    rng = np.random.default_rng(seed)
    k = len(symmetry.orbital_irreps)
    labels = np.asarray(symmetry.orbital_irreps)
    one = rng.normal(size=(k, k))
    one = (one+one.T) * .5
    one[labels[:, None] != labels] = 0.
    p, q = np.tril_indices(k)
    code = labels[p] ^ labels[q]
    pair = rng.normal(size=(len(p), len(p)))
    pair = (pair+pair.T) * .5
    pair[code[:, None] != code] = 0.
    index = np.empty((k, k), dtype=np.intp)
    index[p, q] = index[q, p] = np.arange(len(p))
    two = pair[index[:, None, :, None], index[None, :, None, :]]
    return scsf.Hamiltonian(.125, one, two), pair


def basis(k, n, spin, labels=None, **kwargs):
    wfn = scsf.csf_wfn(k, n, spin, **kwargs)
    if labels is None:
        wfn.add_all_csfs()
    else:
        for label in labels:
            wfn.add_csf(label)
    return wfn


@pytest.mark.parametrize('group,codes', [('C1', [0]*4), ('C2', [0, 1, 0, 1]),
                                          ('D2h', [0, 1, 2, 4, 7, 1])])
def test_block_storage_rotation_and_snapshot(group, codes, tmp_path):
    symmetry = scsf.PointGroup(group, codes)
    dense, fourfold = spatial_integrals(symmetry)
    packed = scsf.PackedHamiltonian.from_dense(dense)
    blocked = scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry)
    np.testing.assert_array_equal(blocked.to_global_packed().packed_eri, packed.packed_eri)
    np.testing.assert_array_equal(scsf.BlockPackedHamiltonian.from_packed(
        packed, symmetry=symmetry).packed_eri, blocked.packed_eri)
    np.testing.assert_array_equal(
        scsf.BlockPackedHamiltonian.from_pair_matrix(dense.ecore, dense.one_mo, fourfold,
                                                    symmetry=symmetry).packed_eri, blocked.packed_eri)
    if group == 'C1':
        np.testing.assert_array_equal(blocked.packed_eri, packed.packed_eri)
    else:
        assert blocked.packed_eri.size < packed.packed_eri.size
    path = tmp_path/'integrals.npz'
    scsf.save_hamiltonian(path, blocked, nelec=4)
    loaded, nelec = scsf.load_hamiltonian(path)
    assert nelec == 4 and loaded.symmetry == symmetry
    np.testing.assert_array_equal(blocked.packed_eri, loaded.packed_eri)
    rotation = natural_orbitals(dense.one_mo, 'Fixture', symmetry=symmetry)[:, ::-1]
    rotated = rotate_hamiltonian(blocked, rotation)
    expected = scsf.PackedHamiltonian.from_dense(rotate_hamiltonian(dense, rotation))
    np.testing.assert_allclose(rotated.to_global_packed().packed_eri, expected.packed_eri, atol=3e-14)
    np.testing.assert_allclose(rotate_hamiltonian(blocked.to_global_packed(), rotation).packed_eri,
                               expected.packed_eri, atol=3e-14)
    np.testing.assert_allclose(rotated.one_mo, expected.one_mo, atol=3e-14)


def test_sector_dimensions_and_basis_round_trip(tmp_path):
    symmetry = scsf.PointGroup('D2h', [0, 1, 6, 3, 0])
    for n, spin in [(0, 0), (3, 1), (4, 0), (4, 2), (10, 0)]:
        full = basis(5, n, spin)
        for code in range(8):
            selected = basis(5, n, spin, symmetry=symmetry, target_irrep=code)
            expected = [label for label in full if symmetry.configuration_irrep(label) == code]
            assert list(selected) == expected
            assert len(expected) == scsf.csf_dimension(5, n, spin, symmetry=symmetry, target_irrep=code)
    selected = basis(5, 4, 0, symmetry=symmetry, target_irrep='B1g')
    with pytest.raises(ValueError, match='target irrep'):
        selected.add_csf('22000')
    path = tmp_path/'labels.json'
    selected.to_file(path)
    loaded = scsf.csf_wfn.from_file(path)
    assert loaded.symmetry == symmetry and loaded.target_irrep == 1
    assert list(loaded) == list(selected)
    empty_group = scsf.PointGroup('C2', [])
    empty_ham = scsf.BlockPackedHamiltonian(0., np.empty((0, 0)), [], symmetry=empty_group)
    empty_space = basis(0, 0, 0, symmetry=empty_group, target_irrep=1)
    assert scsf.csf_structured_op(empty_ham, empty_space, nthreads=1).shape == (0, 0)


def test_union_of_sector_spectra_matches_determinant_reference():
    symmetry = scsf.PointGroup('C2v', [0, 1, 3, 0])
    dense, _ = spatial_integrals(symmetry)
    ham = scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry)
    full = basis(4, 3, 1)
    reference = hamiltonian_block(dense, tuple(full), tuple(full), 1)
    spectra = []
    for code in range(4):
        selected = basis(4, 3, 1, symmetry=symmetry, target_irrep=code)
        indices = [full.index_csf(label) for label in selected]
        op = scsf.csf_structured_op(ham, selected, nthreads=2)
        block = op.matmat(np.eye(len(selected)))
        np.testing.assert_allclose(block, reference[np.ix_(indices, indices)], atol=1e-14, rtol=0)
        spectra.extend(np.linalg.eigvalsh(block))
    np.testing.assert_allclose(np.sort(spectra), np.linalg.eigvalsh(reference), atol=1e-14, rtol=0)


def test_forbidden_integral_validation_and_rotation_opt_out():
    symmetry = scsf.PointGroup('C2', ['A', 'B'])
    dense, pair = spatial_integrals(symmetry)
    forbidden = pair.copy()
    forbidden[0, 1] = 1e-4
    forbidden[1, 0] = -1e-4
    for cls in (scsf.PackedHamiltonian, scsf.BlockPackedHamiltonian):
        with pytest.raises(ValueError, match='forbidden'):
            cls.from_pair_matrix(dense.ecore, dense.one_mo, forbidden, symmetry=symmetry)
        one = dense.one_mo.copy()
        one[0, 1] = one[1, 0] = 2e-12
        with pytest.raises(ValueError, match='forbidden'):
            cls.from_pair_matrix(dense.ecore, one, pair, symmetry=symmetry)
        ham = cls.from_pair_matrix(dense.ecore, one, pair, symmetry=symmetry, symmetry_atol=1e-10)
        assert ham.one_mo[0, 1] == 0.
        rotation = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
        with pytest.raises(ValueError, match='mixes orbital irreps'):
            rotate_hamiltonian(ham, rotation)
        assert rotate_hamiltonian(ham, rotation, preserve_symmetry=False).symmetry is None


@pytest.mark.parametrize('kernel,options', [
    ('nminus2', {'pair_accumulation': 'private'}),
    ('nminus2', {'pair_accumulation': 'gather'}),
    ('nminus1', {}), ('cih', {}), ('cih', {'cih_slot_sweep': True}), ('particle-hole', {}),
])
@pytest.mark.parametrize('target', [None, 0])
def test_signed_products_and_diagonals(kernel, options, target):
    symmetry = scsf.PointGroup('D2h', [0, 1, 2, 4, 7, 1])
    dense, _ = spatial_integrals(symmetry)
    labels = list(basis(6, 4, 0, symmetry=symmetry, target_irrep=target))[::-2]
    selected = basis(6, 4, 0, labels, symmetry=symmetry, target_irrep=target)
    reference = scsf.csf_structured_op(dense, basis(6, 4, 0, labels), strategy='particle-hole', nthreads=1)
    x = np.random.default_rng(51).normal(size=(len(labels), 8))
    np.testing.assert_allclose(reference.diagonal(), np.diag(reference.matmat(np.eye(len(labels)))), atol=3e-14)
    for ham in (scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry),
                scsf.PackedHamiltonian.from_dense(dense, symmetry=symmetry)):
        actual = scsf.csf_structured_op(ham, selected, strategy=kernel, nthreads=3, **options)
        np.testing.assert_allclose(actual.diagonal(), reference.diagonal(), atol=5e-14, rtol=0)
        for width in (1, 2, 8):
            np.testing.assert_allclose(actual.matmat(x[:, :width]), reference.matmat(x[:, :width]),
                                       atol=8e-14, rtol=0)
        probe, _ = actual._state[1]._generator_probe(actual.to_native_order(x[:, 0]),
                                                    2**20, 2**20, 2**20)
        np.testing.assert_allclose(probe, actual.native_matmat(actual.to_native_order(x[:, :1]))[:, 0],
                                   atol=8e-14, rtol=0)


@pytest.mark.parametrize('complete', [False, True])
def test_ordered_and_direct_selection_replay(complete):
    symmetry = scsf.PointGroup('C2v', [0, 1, 2, 0, 3, 1])
    dense, _ = spatial_integrals(symmetry)
    blocked = scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry)
    labels = ['220000', '202000', '000220']
    reference = scsf.csf_structured_op(dense, basis(6, 4, 0, labels), nthreads=2)
    estimates = np.array([.8, .4, .3])
    for ordered in (False, True):
        wfn = basis(6, 4, 0, labels, symmetry=symmetry, target_irrep=0)
        actual = scsf.csf_structured_op(blocked, wfn, nthreads=2, ordered_hci=ordered)
        if not ordered:
            expected = reference
        else:
            expected = scsf.csf_structured_op(dense, basis(6, 4, 0, labels), nthreads=2)
        unit = 'configuration' if complete else 'csf'
        for iteration in range(2):
            size = actual.shape[0]
            weights = estimates if not iteration else np.linspace(0., .9, size)
            remap, stats = actual.select_hci(weights, .1, selection_unit=unit)
            mapped, baseline = expected.select_hci(weights, .1, selection_unit=unit)
            np.testing.assert_array_equal(remap, mapped)
            assert stats['ordered_search'] == ordered
            assert stats['added'] == baseline['added']
            assert list(actual.labels()) == list(expected.labels())
            actual.append_selected_labels(wfn, size)
            assert all(symmetry.configuration_irrep(label) == 0 for label in wfn)


def test_ordered_tables_are_optional_for_limited_packed_growth():
    symmetry = scsf.PointGroup('C2', [0, 1, 0, 0, 1, 1])
    dense, _ = spatial_integrals(symmetry)
    ham = scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry)
    wfn = basis(6, 4, 0, ['220000'], symmetry=symmetry, target_irrep=0)
    direct = scsf.csf_structured_op(ham, wfn, ordered_hci=False, nthreads=1)
    expected_remap, expected_stats = direct.select_hci([1.], .1)
    cap = direct.product_info['peak_tracked_memory_bytes']
    limited = scsf.csf_structured_op(ham, wfn, max_mem=cap, nthreads=1)
    remap, stats = limited.select_hci([1.], .1)
    np.testing.assert_array_equal(remap, expected_remap)
    assert expected_stats['added'] == stats['added'] > 0
    assert list(limited.labels()) == list(direct.labels())
    assert limited.product_info['peak_tracked_memory_bytes'] <= cap


@pytest.mark.parametrize('ordered', [False, True])
def test_symmetry_screening_preserves_strict_selection_threshold(ordered):
    symmetry = scsf.PointGroup('C2v', [0, 1, 0, 2])
    one = np.zeros((4, 4))
    one[0, 2] = one[2, 0] = 1.
    ham = scsf.BlockPackedHamiltonian.from_pair_matrix(0., one, np.zeros((10, 10)), symmetry=symmetry)
    wfn = basis(4, 1, 1, ['u000'], symmetry=symmetry, target_irrep=0)
    op = scsf.csf_structured_op(ham, wfn, ordered_hci=ordered, nthreads=1)
    assert op.select_hci([1.], 1.)[1]['added'] == 0
    assert op.select_hci([1.], np.nextafter(1., 0.))[1]['added'] == 1
    assert list(op.labels()) == ['u000', '00u0']


def test_wide_cih_and_incremental_nminus2():
    k = 65
    active = np.array([0, 27, 63, 64])
    symmetry = scsf.PointGroup('C2v', np.arange(k) % 4)
    small_group = scsf.PointGroup('C2v', [symmetry.orbital_irreps[p] for p in active])
    dense, pair = spatial_integrals(small_group)
    p, q = np.tril_indices(len(active))
    a, b = active[p], active[q]
    pairs = a*(a+1)//2 + b
    fourfold = np.zeros((k*(k+1)//2,)*2)
    fourfold[np.ix_(pairs, pairs)] = pair
    one = np.zeros((k, k))
    one[np.ix_(active, active)] = dense.one_mo
    ham = scsf.BlockPackedHamiltonian.from_pair_matrix(dense.ecore, one, fourfold, symmetry=symmetry)
    short = basis(4, 3, 1)
    frozen = []
    for label in short:
        long = ['0']*k
        for p, step in zip(active, label):
            long[p] = step
        frozen.append(''.join(long))
    wide = basis(k, 3, 1, frozen, symmetry=symmetry)
    ref = scsf.csf_structured_op(dense, short, strategy='particle-hole', nthreads=1)
    actual = scsf.csf_structured_op(ham, wide, strategy='cih', nthreads=2)
    x = np.random.default_rng(14).normal(size=(len(short), 2))
    assert actual.product_info['wide_grouped_uga'] == 1
    np.testing.assert_allclose(actual.matmat(x), ref.matmat(x), atol=6e-14, rtol=0)
    np.testing.assert_allclose(actual.diagonal(), ref.diagonal(), atol=6e-14, rtol=0)
    growing = basis(k, 3, 1, frozen[:-2], symmetry=symmetry)
    incremental = scsf.csf_structured_op(ham, growing, nthreads=2)
    for label in frozen[-2:]:
        growing.add_csf(label)
    incremental.update(ham, growing)
    rebuilt = scsf.csf_structured_op(ham, growing, nthreads=2, incremental_removals=False)
    assert incremental.product_info['incremental_removal_groups'] > 0
    np.testing.assert_allclose(incremental.diagonal(), rebuilt.diagonal(), atol=6e-14, rtol=0)
    np.testing.assert_allclose(incremental.matmat(x), rebuilt.matmat(x), atol=6e-14, rtol=0)
    cap = rebuilt.product_info['peak_tracked_memory_bytes']
    limited = scsf.csf_structured_op(ham, growing, nthreads=2, max_mem=cap)
    np.testing.assert_allclose(limited.matmat(x), rebuilt.matmat(x), atol=6e-14, rtol=0)
    info = limited.product_info
    assert info['peak_tracked_memory_bytes'] <= cap
    assert info['removal_retry_ns'] > 0
    assert sum(value for name, value in info.items()
               if name.startswith('removal_retry_') and name != 'removal_retry_ns') == 1


@pytest.mark.parametrize('threads', [1, 4])
def test_compact_rdms_and_energy_contractions(threads):
    symmetry = scsf.PointGroup('C2v', [0, 1, 2, 2, 3])
    dense, _ = spatial_integrals(symmetry)
    blocked = scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry)
    labels = list(basis(5, 3, 1))[::-1]
    actual = scsf.csf_structured_op(blocked, basis(5, 3, 1, labels, symmetry=symmetry), nthreads=threads)
    baseline = scsf.csf_structured_op(dense, basis(5, 3, 1, labels), nthreads=1)
    codes = np.array([symmetry.configuration_irrep(label) for label in actual.labels(native=True)])
    rng = np.random.default_rng(602)
    x, y = rng.normal(size=(2, len(labels)))
    left, right = x*(codes == 0), y*(codes == 2)
    for bra, ket in ((x, x), (left, left), (left, right), (x, y), (x, right), (x*0, y)):
        rdm1 = actual.native_transition_rdm1(bra, ket)
        rdm2 = actual.native_transition_rdm2(bra, ket)
        block = actual.native_transition_rdm2(bra, ket, layout='blocks')
        np.testing.assert_allclose(rdm1, baseline.native_transition_rdm1(bra, ket), atol=5e-14)
        np.testing.assert_allclose(rdm2, baseline.native_transition_rdm2(bra, ket), atol=5e-14)
        np.testing.assert_allclose(block.to_dense(), rdm2, atol=5e-14)
        projection = block.eri_projection()
        total = np.sum(rdm1*dense.one_mo) + .5*np.dot(blocked.packed_eri, projection)
        np.testing.assert_allclose(total, bra @ actual.native_matmat(ket[:, None])[:, 0], atol=1e-13)
        np.testing.assert_allclose(actual.native_density_energy(bra, ket), total, atol=1e-13)
    one, two, stats = actual.native_rdms(left, layout='blocks', cache=True)
    assert two.transfer_irreps == (0,) and stats['output_bytes'] < (5**2+5**4)*8
    _, _, again = actual.native_rdms(left, layout='blocks', cache=True)
    assert again['cache_hit'] == 1
    _, _, dense_stats = actual.native_rdms(left, cache=True)
    assert dense_stats['cache_hit'] == 0
    np.testing.assert_allclose(np.trace(one), 3*np.dot(left, left))
    with pytest.raises(MemoryError, match='output budget'):
        actual.native_rdms(left, layout='blocks', max_output_bytes=stats['output_bytes']-1)


def test_response_keeps_reference_sector_when_property_adds_lower_root():
    symmetry = scsf.PointGroup('C2', [0, 1, 0])
    ham = scsf.BlockPackedHamiltonian(0., np.diag([-1., -2., 1.]), np.zeros(13), symmetry=symmetry)
    original = basis(3, 1, 1, ['u00'], symmetry=symmetry, target_irrep=0)
    operator = scsf.csf_structured_op(ham, original, nthreads=1)
    prop = np.zeros((3, 3), complex)
    prop[1, 0] = 2j
    prop[2, 0] = 1
    perturbation = ('a', .2, 1)
    wfn, op, vectors, _, _, response = resp(ham, original, operator, np.ones((1, 1)), {'a': prop},
                                           [perturbation], gamma=.05, eps_mu=.01, verbose=False)
    assert set(wfn) == {'u00', '0u0', '00u'}
    assert list(original) == ['u00'] and op.symmetry == symmetry
    assert np.count_nonzero(vectors[:, 0]) == 1
    expected = 4./(-1.2-.05j) + 1./(1.8-.05j)
    assert response['a', 'a', .2] == pytest.approx(expected, rel=2e-8)


def test_molecular_target_and_natural_orbital_driver(capsys):
    pytest.importorskip('pyscf')
    run_sci(['--xyz', 'examples/water.xyz', '--basis', 'sto-3g', '--target-irrep', 'B1',
             '--natorb', 'sci', '--eps', '.001', '--nthreads', '1', '--quiet'])
    output = capsys.readouterr().out
    assert output.count('Point group C2v, seed irreps B1') == 2
    root = next(line for line in output.splitlines() if line.startswith('root=0 '))
    assert float(root.split('energy=')[1].split()[0]) == pytest.approx(-74.556340755, abs=2e-7, rel=0)
