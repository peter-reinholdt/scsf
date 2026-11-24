"""Signed and memory-bounded CSF Epstein–Nesbet corrections."""

import numpy as np
import pytest

import scsf
from scsf.drivers import csf_driver
from .reference.oracle import hamiltonian_block, random_hamiltonian
from .test_point_group import basis, spatial_integrals


def exact_correction(ham, selected, external, spin, energies, coefficients, eps):
    elements = hamiltonian_block(ham, external, selected, spin)
    terms = elements[:, :, None]*coefficients[None, :, :]
    t = np.sum(np.where(np.abs(terms)>eps, terms, 0.), axis=1)
    diagonal = np.diag(hamiltonian_block(ham, external, external, spin))
    d = energies[None, :]-ham.ecore-diagonal[:, None]
    mask = t != 0.
    minimum = np.min(np.where(mask, np.abs(d), np.inf), axis=0)
    return (np.sum(np.divide(t*t, d, out=np.zeros_like(t), where=mask), axis=0),
            minimum, np.sum(mask & (d>0.), axis=0))


@pytest.mark.parametrize('kernel', scsf.csf_structured_op.kernels)
@pytest.mark.parametrize('eps', [0., .12])
def test_state_specific_correction_for_partial_configurations(kernel, eps):
    dense = random_hamiltonian(5, seed=45)
    ham = scsf.Hamiltonian(1.375, dense.one_mo, dense.two_mo)
    labels = ['uudd0', '22000', 'uu0dd', 'udud0', '0u2d0', '2u0d0']
    wfn = basis(5, 4, 0, labels)
    external = [a for a in basis(5, 4, 0) if a not in labels]
    e, v = np.linalg.eigh(hamiltonian_block(ham, labels, labels, 0))
    energies, vectors = e[[0, 2]]+ham.ecore, v[:, [0, 2]]
    expected, minimum, positive = exact_correction(ham, labels, external, 0, energies, vectors, eps)

    op = scsf.csf_structured_op(ham, wfn, strategy=kernel, nthreads=2)
    assert tuple(op.labels()) != tuple(op.labels(native=True))
    result = op.enpt2(energies, vectors, eps=eps, screening='strict')
    assert isinstance(result, scsf.ENPT2Result)
    np.testing.assert_allclose(result.corrections, expected, atol=1e-11, rtol=0)
    np.testing.assert_allclose(result.minimum_abs_denominators, minimum, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(result.positive_denominator_counts, positive)
    np.testing.assert_array_equal(result.corrected_energies, energies+result.corrections)
    native = op.native_enpt2(energies, op.to_native_order(vectors), eps=eps, screening='strict')
    np.testing.assert_array_equal(native.corrections, result.corrections)
    for r in range(energies.size):
        one = op.enpt2(energies[r], vectors[:, r], eps=eps, screening='strict')
        np.testing.assert_allclose(one.corrections, expected[r:r+1], atol=1e-11, rtol=0)


@pytest.mark.parametrize('cutoff', [0., .09])
def test_labeled_layouts_and_mixed_sectors(cutoff):
    symmetry = scsf.PointGroup('C2', [0, 1, 1, 0, 0])
    dense, _ = spatial_integrals(symmetry)
    block_packed = scsf.BlockPackedHamiltonian.from_dense(dense, symmetry=symmetry)
    packed = block_packed.to_global_packed(symmetry=True)
    dense_labeled = scsf.Hamiltonian(dense.ecore, dense.one_mo, dense.two_mo, symmetry=symmetry)
    labels = ['uu0dd', '22000', 'u2d00', '0u2d0', 'ud020', '0u0d2']
    external = [a for a in basis(5, 4, 0) if a not in labels]
    e, v = np.linalg.eigh(hamiltonian_block(dense, labels, labels, 0))
    energies = e[:2]+dense.ecore
    expected, _, _ = exact_correction(dense, labels, external, 0, energies, v[:, :2], cutoff)
    for ham in (dense_labeled, packed, block_packed):
        for ordered in (False, True):
            op = scsf.csf_structured_op(ham, basis(5, 4, 0, labels, symmetry=symmetry),
                                       nthreads=2, ordered_hci=ordered)
            actual = op.enpt2(energies, v[:, :2], eps=cutoff, screening='strict')
            np.testing.assert_allclose(actual.corrections, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize('path,spin', [('h4_dalton_sto3g', 0), ('h4_dalton_sto3g', 2), ('lih_sto3g', 0)])
def test_molecular_reference_projection(path, spin):
    ham, nelec = scsf.load_hamiltonian(f'scsf/test/data/{path}.npz')
    all_labels = list(basis(ham.nbasis, nelec, spin))
    labels, external = all_labels[::2], all_labels[1::2]
    e, v = np.linalg.eigh(hamiltonian_block(ham, labels, labels, spin))
    energies, vectors = e[:2]+ham.ecore, v[:, :2]
    expected, _, _ = exact_correction(ham, labels, external, spin, energies, vectors, 0.)
    op = scsf.csf_structured_op(ham, basis(ham.nbasis, nelec, spin, labels), strategy='cih', nthreads=2)
    np.testing.assert_allclose(op.enpt2(energies, vectors, screening='strict').corrections,
                               expected, atol=1e-11, rtol=0)
    np.testing.assert_allclose(op.enpt2(energies, vectors, screening='block').corrections,
                               expected, atol=1e-11, rtol=0)


def test_signed_interference_cutoff_and_denominator_guard():
    one = np.array([[0., 0., .125], [0., 0., .125], [.125, .125, -2.]])
    ham = scsf.Hamiltonian(.5, one, np.zeros((3,)*4))
    wfn = basis(3, 1, 1, ['u00', '0u0'])
    op = scsf.csf_structured_op(ham, wfn, nthreads=1)
    v = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2.)
    result = op.enpt2([.5, .5], v)
    assert result.corrections == pytest.approx([0., .015625], abs=1e-15)
    assert result.positive_denominator_counts.tolist() == [0, 1]
    assert np.isinf(result.minimum_abs_denominators[0])
    assert result.minimum_abs_denominators[1] == 2.
    assert op.enpt2(.5, [1., 0.], eps=.125, screening='strict').corrections[0] == 0.
    assert op.enpt2(.5, [1., 0.], eps=np.nextafter(.125, 0.), screening='strict').corrections[0] == .0078125

    one[2, 2] = 0.
    degenerate = scsf.csf_structured_op(scsf.Hamiltonian(.5, one, np.zeros((3,)*4)), wfn, nthreads=1)
    assert degenerate.enpt2(.5, v[:, 0]).corrections[0] == 0.
    with pytest.raises(ValueError, match='root 0, CSF 00u.*denominator=0'):
        degenerate.enpt2(.5, [1., 0.], denominator_tol=0.)
    assert degenerate.enpt2(.5, v[:, 0]).corrections[0] == 0.
    one[2, 2] = 5e-11
    close = scsf.csf_structured_op(scsf.Hamiltonian(.5, one, np.zeros((3,)*4)), wfn, nthreads=1)
    with pytest.raises(ValueError, match='denominator within tolerance'):
        close.enpt2(.5, [1., 0.])
    assert close.enpt2(.5, [1., 0.], denominator_tol=1e-12).corrections[0] == -.125**2/5e-11


def test_whole_block_screening_includes_weak_signed_terms():
    ham = random_hamiltonian(5, seed=45)
    labels = ['uudd0', 'udud0']
    external = [a for a in basis(5, 4, 0) if a not in labels]
    elements = hamiltonian_block(ham, external, labels, 0)
    vector = np.array([1., -1.e-5])
    vector /= np.linalg.norm(vector)
    maxima = {}
    for label, row in zip(external, elements):
        occ = label.translate(str.maketrans('ud', '11'))
        maxima[occ] = max(maxima.get(occ, 0.), np.max(np.abs(row*vector)))
    cutoff = .999*min(value for value in maxima.values() if value > 0.)
    energy = np.array([-30.])
    expected, _, _ = exact_correction(ham, labels, external, 0, energy, vector[:, None], 0.)
    op = scsf.csf_structured_op(ham, basis(5, 4, 0, labels), strategy='nminus2', nthreads=1)
    strict = op.enpt2(energy, vector, eps=cutoff, screening='strict')
    block = op.enpt2(energy, vector, eps=cutoff)
    assert block.stats['block_contractions'] > 0
    assert abs(block.corrections[0]-strict.corrections[0]) > 1e-9
    np.testing.assert_allclose(block.corrections, expected, atol=1e-12, rtol=0)
    op = scsf.csf_structured_op(ham, basis(5, 4, 0, labels+['22000']), strategy='nminus2', nthreads=1)
    vectors = np.array([[1., 0.], [0., 0.], [0., 1.]])
    both = op.enpt2([-30., -31.], vectors, eps=.2, screening='block')
    separate = [op.enpt2(e, v, eps=.2, screening='block').corrections[0]
                for e, v in zip([-30., -31.], vectors.T)]
    np.testing.assert_allclose(both.corrections, separate, atol=1e-13, rtol=0)


def test_bounded_batches_stream_oversized_targets_and_preserve_operator():
    ham = random_hamiltonian(6, seed=32)
    all_labels = list(basis(6, 4, 0))
    labels = all_labels[1::2]
    wfn = basis(6, 4, 0, labels)
    e, v = np.linalg.eigh(hamiltonian_block(ham, labels, labels, 0))
    energy, vector = e[0]+ham.ecore, v[:, 0]
    wide = scsf.csf_structured_op(ham, wfn, nthreads=4, ordered_hci=False)
    expected = wide.enpt2(energy, vector)
    assert expected.stats['diagonal_factor_bytes'] > 0
    budget = expected.stats['minimum_tracked_memory_bytes']
    bounded = scsf.csf_structured_op(ham, wfn, nthreads=4, ordered_hci=False, max_mem=budget)
    actual = bounded.enpt2(energy, vector, screening='strict')
    assert actual.stats['batch_splits'] and actual.stats['streamed_targets']
    assert actual.stats['peak_tracked_memory_bytes'] <= budget
    assert actual.stats['rank_tile_rows'] == 1
    assert actual.stats['diagonal_factor_bytes'] == 0
    np.testing.assert_allclose(actual.corrections, expected.corrections, atol=1e-11, rtol=0)
    np.testing.assert_array_equal(actual.contributing_csfs, expected.contributing_csfs)
    np.testing.assert_allclose(actual.minimum_abs_denominators, expected.minimum_abs_denominators,
                               atol=1e-12, rtol=0)
    block = bounded.enpt2(energy, vector, screening='block')
    assert block.stats['block_factor_bytes'] == 0
    assert block.stats['peak_tracked_memory_bytes'] <= budget
    np.testing.assert_allclose(block.corrections, expected.corrections, atol=1e-11, rtol=0)
    insufficient = scsf.csf_structured_op(ham, wfn, nthreads=4, ordered_hci=False, max_mem=budget-1)
    with pytest.raises(MemoryError, match='ENPT2 minimum worker workspace'):
        insufficient.enpt2(energy, vector)
    np.testing.assert_allclose(insufficient.matvec(vector), (energy-ham.ecore)*vector, atol=1e-11, rtol=0)


@pytest.mark.parametrize('k', [66, 80, 81, 130])
def test_wide_occupation_key_and_full_variational_space(k):
    one = np.diag(np.arange(k, dtype=float))
    one[0, k-1] = one[k-1, 0] = .5
    one[0, 1] = one[1, 0] = .25
    pairs = k*(k+1)//2
    ham = scsf.PackedHamiltonian(2., one, np.zeros(pairs*(pairs+1)//2))
    op = scsf.csf_structured_op(ham, basis(k, 1, 1, ['u'+'0'*(k-1)]), nthreads=1)
    result = op.enpt2(2., [1.])
    assert result.corrections[0] == pytest.approx(-.25/(k-1)-.0625, abs=1e-15)
    assert result.contributing_csfs[0] == 2
    op = scsf.csf_structured_op(ham, basis(k, 1, 1), nthreads=2)
    energies, vectors = np.linalg.eigh(one)
    result = op.enpt2(energies[0]+ham.ecore, vectors[:, 0])
    assert result.corrections[0] == 0. and result.contributing_csfs[0] == 0
    with pytest.raises(ValueError, match='normalized'):
        op.enpt2(2., np.zeros(k))
    with pytest.raises(TypeError, match='real'):
        op.enpt2(2., vectors[:, 0].astype(complex))


@pytest.mark.parametrize('k', [64, 65])
def test_double_transfers_with_closed_spectators_at_word_boundary(k):
    dense = random_hamiltonian(5, seed=45)
    compact = scsf.PackedHamiltonian.from_dense(dense)
    labels = ['uudd0', '22000', 'uu0dd', '0u2d0', '2u0d0']
    external = [label for label in basis(5, 4, 0) if label not in labels]
    e, v = np.linalg.eigh(hamiltonian_block(dense, labels, labels, 0))
    energies, vectors = e[:2]+dense.ecore, v[:, :2]
    expected, _, _ = exact_correction(dense, labels, external, 0, energies, vectors, .12)

    active = np.array([0, 7, 28, 47, k-1])
    closed = [10, 48, 60]
    one = np.zeros((k, k))
    one[np.ix_(active, active)] = dense.one_mo
    pairs = k*(k+1)//2
    packed = np.zeros(pairs*(pairs+1)//2)
    p, q = np.tril_indices(len(active))
    occupied_pairs = active[p]*(active[p]+1)//2+active[q]
    p, q = np.tril_indices(len(occupied_pairs))
    packed[occupied_pairs[p]*(occupied_pairs[p]+1)//2+occupied_pairs[q]] = compact.packed_eri
    ham = scsf.PackedHamiltonian(dense.ecore, one, packed)

    def embed(label):
        result = ['0']*k
        for orbital in closed:
            result[orbital] = '2'
        for orbital, step in zip(active, label):
            result[orbital] = step
        return ''.join(result)

    wfn = basis(k, 4+2*len(closed), 0, map(embed, labels))
    op = scsf.csf_structured_op(ham, wfn, strategy='particle-hole', nthreads=1, ordered_hci=False)
    actual = op.enpt2(energies, vectors, eps=.12, screening='strict')
    np.testing.assert_allclose(actual.corrections, expected, atol=1e-11, rtol=0)
    block = op.enpt2(energies, vectors, screening='block')
    expected, minimum, positive = exact_correction(dense, labels, external, 0, energies, vectors, 0.)
    assert block.stats['block_contractions'] > 0
    np.testing.assert_allclose(block.corrections, expected, atol=1e-11, rtol=0)
    np.testing.assert_allclose(block.minimum_abs_denominators, minimum, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(block.positive_denominator_counts, positive)


def test_adaptive_prefix_plan_matches_serial_screened_correction():
    labels = [label+'0' for label in basis(7, 6, 0)]
    wfn = basis(8, 6, 0, labels)
    one = np.zeros((8, 8))
    one[7, 7] = 1.
    one[0, 7] = one[7, 0] = .125
    ham = scsf.Hamiltonian(0., one, np.zeros((8,)*4))
    vector = np.random.default_rng(49).normal(size=len(wfn))
    vector /= np.linalg.norm(vector)
    serial = scsf.csf_structured_op(ham, wfn, nthreads=1, ordered_hci=False)
    budget = 16*1024**2
    adaptive = scsf.csf_structured_op(ham, wfn, nthreads=8, max_mem=budget)
    expected = serial.enpt2(0., vector, eps=.001, screening='strict')
    result = adaptive.enpt2(0., vector, eps=.001, screening='strict')
    assert result.stats['adaptive_prefixes']
    assert result.stats['initial_tasks'] > 1
    assert result.stats['ordered_doubles'] and result.stats['ordered_singles']
    assert result.stats['peak_tracked_memory_bytes'] <= budget
    assert result.corrections[0] < 0.
    np.testing.assert_allclose(result.corrections, expected.corrections, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(result.contributing_csfs, expected.contributing_csfs)


@pytest.mark.parametrize('natural', [False, True])
def test_cli_enpt2_runs_only_on_the_final_basis(natural, monkeypatch, capsys):
    original = scsf.csf_structured_op.native_enpt2
    seen = []

    def capture(op, energies, vectors, **options):
        result = original(op, energies, vectors, **options)
        seen.append((op.shape[0], options, result))
        return result

    monkeypatch.setattr(scsf.csf_structured_op, 'native_enpt2', capture)
    screening = 'block' if natural else 'strict'
    csf_driver.main(['--integrals', 'scsf/test/data/lih_sto3g.npz', '--eps', '.05',
                     '--enpt2', '--enpt2-eps', '.0001', '--nthreads', '2', '--quiet']
                    + (['--enpt2-screening', 'strict'] if not natural else [])
                    + (['--natorb', 'sci'] if natural else []))
    output = capsys.readouterr().out
    assert len(seen) == 1
    ncsf, options, result = seen[0]
    assert options['eps'] == .0001
    assert options['screening'] == screening
    assert f'ncsf={ncsf} energies=' in [line for line in output.splitlines() if line.startswith('iteration=')][-1]
    assert f'root=0 energy={result.variational_energies[0]:.12f}' in output
    assert f'root=0 enpt2={result.corrections[0]:.12f} corrected_energy={result.corrected_energies[0]:.12f}' in output
    assert 'Epstein–Nesbet PT2' in output
