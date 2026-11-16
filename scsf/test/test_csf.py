# SCSF; licensed under GPL-3.0-or-later.
from scsf.test import load_hamiltonian
"""Independent spin/operator checks for the transformed-determinant backend."""
import copy
from itertools import combinations, product
import json
from math import comb
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf.csf import connected_configurations
from scsf.test import datafile
from scsf.drivers.solvers import solve_ci

def davidson(op, n=1, c0=None, ncv=None):
    """Exercise the external Davidson solver, returning rows for these tests."""
    if hasattr(op, 'matmat'):
        hvp = op.matmat
    else:
        hvp = lambda block: np.column_stack([op(v) for v in block.T])
    if c0 is not None and np.asarray(c0).ndim == 1:
        c0 = np.asarray(c0)[:, None]
    energies, coeffs, converged = solve_ci(hvp, op.diagonal(), roots=n, c0=c0, tol=1e-12, maxiter=300, max_subspace=None if ncv is None else min(ncv, op.shape[0]))
    assert converged
    return (energies + op.ecore, coeffs.T)

def random_hamiltonian(k=4, seed=17):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(k, k))
    h = (h + h.T) / 2
    g = rng.normal(size=(k,) * 4)
    g = (g + g.swapaxes(0, 1)) / 2
    g = (g + g.swapaxes(2, 3)) / 2
    g = (g + g.transpose(2, 3, 0, 1)) / 2
    return scsf.hamiltonian(1.2345, h, g.transpose(0, 2, 1, 3))

def determinant_keys(wfn):
    words = wfn.to_det_array()
    wordbits = np.dtype(np.uint64).itemsize * 8
    result = []
    for row in words:
        spins = [sum((int(x) << wordbits * i for i, x in enumerate(s))) for s in row]
        result.append(spins[0] | spins[1] << wfn.nbasis)
    return result

def fock_action(occupied, annihilate, create):
    """Independent list-based fermionic action, with creation-order signs."""
    occupied = list(occupied)
    sign = 1
    for p in annihilate:
        if p not in occupied:
            return (None, 0)
        sign *= (-1) ** occupied.index(p)
        occupied.remove(p)
    for p in create:
        if p in occupied:
            return (None, 0)
        sign *= (-1) ** sum((q < p for q in occupied))
        occupied.append(p)
        occupied.sort()
    return (tuple(occupied), sign)

def fock_reference(ham, nelec, two_ms):
    """Build H and S² by explicit second-quantized actions (no SC rules)."""
    k = ham.nbasis
    na, nb = ((nelec + two_ms) // 2, (nelec - two_ms) // 2)
    dets = [a + tuple((p + k for p in b)) for a in combinations(range(k), na) for b in combinations(range(k), nb)]
    lookup = {d: i for i, d in enumerate(dets)}
    h = np.zeros((len(dets), len(dets)))
    s2 = np.eye(len(dets)) * (two_ms / 2) * (two_ms / 2 + 1)
    for j, det in enumerate(dets):
        for p, q, sigma in product(range(k), range(k), range(2)):
            target, phase = fock_action(det, (q + sigma * k,), (p + sigma * k,))
            if phase:
                h[lookup[target], j] += phase * ham.one_mo[p, q]
        terms = product(range(k), range(k), range(k), range(k), range(2), range(2))
        for p, r, q, s, sigma, tau in terms:
            target, phase = fock_action(det, (q + sigma * k, s + tau * k), (r + tau * k, p + sigma * k))
            if phase:
                h[lookup[target], j] += 0.5 * phase * ham.two_mo[p, r, q, s]
        for q, p in product(range(k), repeat=2):
            middle, sign1 = fock_action(det, (q + k,), (q,))
            if sign1:
                target, sign2 = fock_action(middle, (p,), (p + k,))
                if sign2:
                    s2[lookup[target], j] += sign1 * sign2
    keys = [sum((1 << p for p in det)) for det in dets]
    return (keys, h, s2)

def transform_in_order(wfn, keys, two_ms):
    dwfn, u = wfn.determinant_transform(two_ms=two_ms)
    u = u.toarray()
    lookup = {key: i for i, key in enumerate(keys)}
    result = np.zeros((len(keys), len(wfn)))
    for i, key in enumerate(determinant_keys(dwfn)):
        result[lookup[key]] = u[i]
    return result

@pytest.mark.parametrize('k,n,s', [(k, n, s) for k in range(6) for n in range(2 * k + 1) for s in range(n % 2, min(n, 2 * k - n) + 1, 2)])
def test_dimensions(k, n, s):
    w = scsf.csf_wfn(k, n, s)
    w.add_all_csfs()
    a, b = ((n + s) // 2, (n - s) // 2)
    expected = comb(k, a) * comb(k, b)
    if a < k and b:
        expected -= comb(k, a + 1) * comb(k, b - 1)
    assert len(w) == scsf.csf_dimension(k, n, s) == expected
    assert w.ncsf == len(w) and len(set(w)) == len(w)
    assert w.to_step_array().shape == (len(w), k)
    assert w.to_configuration_array().shape == (len(w), k)

@pytest.mark.parametrize('n,s,m', [(0, 0, 0), (2, 0, 0), (2, 2, 0), (2, 2, 2), (3, 1, 1), (3, 3, 1), (3, 3, 3), (4, 0, 0), (4, 2, 0), (4, 2, 2), (4, 4, 0), (4, 4, 2), (4, 4, 4)])
def test_spin_and_signed_matrix_elements(n, s, m):
    ham = random_hamiltonian()
    keys, h, s2 = fock_reference(ham, n, m)
    w = scsf.csf_wfn(4, n, s)
    w.add_all_csfs()
    u = transform_in_order(w, keys, m)
    npt.assert_allclose(u.T @ u, np.eye(len(w)), rtol=0, atol=1e-14)
    npt.assert_allclose(s2 @ u, s / 2 * (s / 2 + 1) * u, rtol=0, atol=1e-14)
    op = scsf.csf_op(ham, w)
    reference = u.T @ h @ u
    npt.assert_allclose(op.to_scipy().toarray(), reference, rtol=0, atol=2e-14)
    npt.assert_allclose(h @ u, u @ reference, rtol=0, atol=2e-14)
    npt.assert_allclose(davidson(op, n=len(w))[0], np.linalg.eigvalsh(reference) + ham.ecore, rtol=0, atol=2e-14)

def test_two_spin_phases_and_six_open_shells():
    singlet = scsf.csf_wfn(2, 2, 0)
    singlet.add_csf('ud')
    d, c = singlet.to_determinants([1.0])
    npt.assert_allclose(c, [1 / np.sqrt(2)] * 2)
    assert set(determinant_keys(d)) == {1 | 2 << 2, 2 | 1 << 2}
    mixed = scsf.csf_wfn(3, 4, 0)
    mixed.add_csf('u2d')
    _, c = mixed.to_determinants([1.0])
    npt.assert_allclose(c, [1 / np.sqrt(2)] * 2)
    mixed = scsf.csf_wfn(3, 4, 0)
    mixed.add_csf('2ud')
    _, c = mixed.to_determinants([1.0])
    npt.assert_allclose(c, [-1 / np.sqrt(2)] * 2)
    w = scsf.csf_wfn(6, 6, 0)
    assert w.add_configuration([1] * 6) == 5
    keys, _, s2 = fock_reference(scsf.hamiltonian(0.0, np.zeros((6, 6)), np.zeros((6,) * 4)), 6, 0)
    u = transform_in_order(w, keys, 0)
    npt.assert_allclose(u.T @ u, np.eye(5), atol=1e-14)
    npt.assert_allclose(s2 @ u, 0, atol=1e-14)

@pytest.mark.parametrize('args', [(-1, 0, 0), (2, -1, 0), (2, 5, 1), (2, 2, 1), (2, 2, 4), (3, 3, -1)])
def test_bad_sectors(args):
    with pytest.raises(ValueError):
        scsf.csf_wfn(*args)

def test_labels_basis_file_and_word_boundary(tmp_path):
    w = scsf.csf_wfn(4, 4, 0)
    for label in ('dddd', 'uuuu', 'ud00', 'u0dx', 'udud0', 'duud'):
        with pytest.raises(ValueError):
            w.add_csf(label)
    assert w.add_csf('uudd') == 0
    assert w.add_csf('uudd') == 0
    w.add_csf('2200')
    w.add_csf('udud')
    assert w.configuration_indices([1] * 4) == (0, 2)
    assert w.index_csf('0022') == -1
    for cloned in (copy.copy(w), copy.deepcopy(w)):
        cloned.add_csf('0022')
        assert len(w) == 3
    path = tmp_path / 'basis.json'
    w.to_file(path)
    assert tuple(scsf.csf_wfn.from_file(path)) == tuple(w)
    data = json.loads(path.read_text())
    data['labels'].append(data['labels'][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='duplicate'):
        scsf.csf_wfn.from_file(path)
    with pytest.raises(AttributeError):
        w.two_s = 2
    big = scsf.csf_wfn(130, 4, 0)
    big.add_csf('u2' + '0' * 127 + 'd')
    d, u = big.determinant_transform()
    npt.assert_allclose((u.T @ u).toarray(), [[1.0]])
    assert all((key.bit_length() > 130 for key in determinant_keys(d)))

def test_matrix_update_rectangular_and_core():
    ham = random_hamiltonian()
    full = scsf.csf_wfn(4, 4, 0)
    full.add_all_csfs()
    w = scsf.csf_wfn(4, 4, 0)
    w.add_csf('uudd')
    lower, general = (scsf.csf_op(ham, w), scsf.csf_op(ham, w, symmetric=False))
    w.add_csf('2200')
    w.add_csf('udud')
    for op in (lower, general):
        op.update(ham, w)
        rebuilt = scsf.csf_op(ham, w, symmetric=op.symmetric)
        npt.assert_array_equal(op.to_scipy().toarray(), rebuilt.to_scipy().toarray())
        before = op.data()
        op.update(ham, w)
        npt.assert_array_equal(op.data(), before)
        x = np.array([0.3, -0.1, 0.7])
        y = op(x)
        npt.assert_allclose(y, op.to_scipy() @ x, atol=1e-14)
        op(x, out=x)
        npt.assert_allclose(x, y)
        for i, j in product(range(3), repeat=2):
            assert op.get_element(i, j) == op.to_scipy()[i, j]
        npt.assert_allclose(davidson(op, n=3)[0] - ham.ecore, np.linalg.eigvalsh(op.to_scipy().toarray()), atol=1e-14)
    reference = scsf.csf_op(ham, full).to_scipy().toarray()
    rect = scsf.csf_op(ham, full, nrow=7, ncol=13, symmetric=False)
    npt.assert_allclose(rect.to_scipy().toarray(), reference[:7, :13], atol=1e-14)
    with pytest.raises(ValueError):
        scsf.csf_op(ham, full, nrow=7, ncol=13)
    with pytest.raises(ValueError):
        rect.update(ham, full)
    empty = scsf.csf_op(ham, scsf.csf_wfn(4, 4, 0))
    assert empty.shape == (0, 0) and empty.size == 0
    npt.assert_array_equal(empty([]), [])
    with pytest.raises(ValueError):
        davidson(empty)

def test_no_pruning_and_missing_entries():
    h = np.array([[0.0, 1e-13, 0.0], [1e-13, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ham = scsf.hamiltonian(3.0, h, np.zeros((3,) * 4))
    w = scsf.csf_wfn(3, 1, 1)
    for label in ('u00', '0u0', '00u'):
        w.add_csf(label)
    op = scsf.csf_op(ham, w, matrix_cutoff=0.0)
    assert op.get_element(0, 1) == 1e-13
    assert op.get_element(0, 2) == 0
    assert op.size == 4
    with pytest.raises(IndexError):
        op.get_element(3, 0)
    assert scsf.add_csf_hci(ham, w, [1, 0, 0], eps=0) == 0

def test_configuration_connectivity_exhaustive():
    configs = [c for c in product(range(3), repeat=5) if sum(c) == 5]
    for source in configs:
        expected = {c for c in configs if sum((abs(a - b) for a, b in zip(source, c))) <= 4}
        assert set(connected_configurations(source)) == expected
    assert list(connected_configurations(())) == [()]

@pytest.mark.parametrize('eps', [0.0, 0.1, 0.5, 2.0, 100.0])
def test_hci_against_independent_exhaustive_criterion(eps):
    ham = random_hamiltonian(seed=291)
    full = scsf.csf_wfn(4, 4, 0)
    full.add_all_csfs()
    keys, h, _ = fock_reference(ham, 4, 0)
    u = transform_in_order(full, keys, 0)
    matrix = u.T @ h @ u
    w = scsf.csf_wfn(4, 4, 0)
    for label in ('uudd', '2200', '02ud'):
        w.add_csf(label)
    parents = [full.index_csf(s) for s in w]
    coeffs = np.array([0.7, -0.3, 0.0])
    expected = {full[i] for i in range(len(full)) if i not in parents and np.max(np.abs(matrix[i, parents] * coeffs)) > eps}
    original = tuple(w)
    added = scsf.add_csf_hci(ham, w, coeffs, eps=eps)
    assert set(w) - set(original) == expected
    assert tuple(w)[len(original):] == tuple(sorted(expected))
    assert added == len(expected)

def test_same_configuration_selection_and_threshold_boundary():
    ham = random_hamiltonian()
    backend = scsf.CSFReferenceBackend(ham)
    element = backend.element('udud', 'uudd', 0)
    assert abs(element) > 0.0001
    w = scsf.csf_wfn(4, 4, 0)
    w.add_csf('uudd')
    scsf.add_csf_hci(ham, w, [1.0], eps=abs(element) / 2, backend=backend)
    assert w.index_csf('udud') >= 0
    ham = scsf.hamiltonian(0.0, np.array([[0.0, 0.125], [0.125, 0.0]]), np.zeros((2,) * 4))
    for eps, expected in ((0.125, 0), (0.125 - 1e-15, 1), (0.125 + 1e-15, 0)):
        w = scsf.csf_wfn(2, 1, 1)
        w.add_csf('u0')
        assert scsf.add_csf_hci(ham, w, [1.0], eps=eps) == expected

def test_guards_validation_and_atomic_selection():
    w = scsf.csf_wfn(4, 4, 0)
    with pytest.raises(MemoryError):
        w.add_all_csfs(max_csfs=19)
    assert len(w) == 0
    w.add_csf('uudd')
    with pytest.raises(MemoryError):
        w.determinant_transform(max_expansion_nnz=1)
    with pytest.raises(MemoryError):
        w.determinant_transform(max_determinants=1)
    for ms in (-2, 1, 2):
        with pytest.raises(ValueError):
            w.determinant_transform(two_ms=ms)
    ham = random_hamiltonian()
    for kw in ({'max_candidates': 0}, {'max_csfs': 1}):
        with pytest.raises(MemoryError):
            scsf.add_csf_hci(ham, w, [1.0], eps=0, **kw)
        assert tuple(w) == ('uudd',)
    for coeffs in ([np.nan], [1.0, 2.0], [[1.0]]):
        with pytest.raises(ValueError):
            scsf.add_csf_hci(ham, w, coeffs)
    with pytest.raises(TypeError):
        scsf.add_csf_hci(ham, w, [1j])
    for eps in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            scsf.add_csf_hci(ham, w, [1.0], eps=eps)
    backend = scsf.CSFReferenceBackend(ham, max_block_elements=1)
    with pytest.raises(MemoryError):
        backend.element('uudd', 'uudd', 0)
    with pytest.raises(MemoryError):
        scsf.csf_op(ham, w, max_matrix_entries=0)
    with pytest.raises(MemoryError):
        scsf.add_csf_hci(ham, w, [1.0], max_configurations=1)
    assert tuple(w) == ('uudd',)
    assert scsf.add_csf_hci(ham, w, [0.0], eps=0) == 0
    with pytest.raises(ValueError):
        scsf.csf_op(ham, w, backend=scsf.CSFReferenceBackend(random_hamiltonian()))

def test_molecular_determinant_and_converted_energy():
    ham = load_hamiltonian(datafile('lih_sto3g.npz'))
    w = scsf.csf_wfn(ham.nbasis, 4, 0)
    w.add_all_csfs()
    op = scsf.csf_op(ham, w)
    energies, coeffs = davidson(op)
    det, dc = w.to_determinants(coeffs[0])
    keys, hdet, _ = fock_reference(ham, 4, 0)
    u = transform_in_order(w, keys, 0)
    dc_ordered = u @ coeffs[0]
    npt.assert_allclose(dc_ordered @ hdet @ dc_ordered+ham.ecore, energies[0], atol=1e-12, rtol=0)
    npt.assert_allclose(np.linalg.norm(dc), 1., atol=1e-13)


@pytest.mark.parametrize('s,reference', [(0, [-2.139442550806343, -1.437749667202313, -1.285223099345084]), (2, [-1.728071475885781, -1.382792978743925, -0.88769257825586])])
def test_dalton_spin_adapted_roots(s, reference):
    ham = load_hamiltonian(datafile('h4_dalton_sto3g.npz'))
    w = scsf.csf_wfn(4, 4, s)
    w.add_all_csfs()
    energies, _ = davidson(scsf.csf_op(ham, w), n=3)
    npt.assert_allclose(energies, reference, rtol=0, atol=1e-08)
    npt.assert_allclose(energies - energies[0], np.array(reference) - reference[0], rtol=0, atol=1e-10)

def test_sparse_eigensolver_and_integral_validation():
    ham = random_hamiltonian(k=5)
    w = scsf.csf_wfn(5, 4, 0)
    w.add_all_csfs()
    assert len(w) == 50
    op = scsf.csf_op(ham, w)
    energies, vectors = davidson(op, n=3, c0=np.ones(len(w)), ncv=20)
    reference = np.linalg.eigvalsh(op.to_scipy().toarray())[:3] + ham.ecore
    npt.assert_allclose(energies, reference, rtol=0, atol=2e-13)
    for energy, vector in zip(energies, vectors):
        npt.assert_allclose(op(vector), (energy - ham.ecore) * vector, rtol=0, atol=1e-11)
    bad_ham = scsf.hamiltonian(0.0, np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((2,) * 4))
    with pytest.raises(ValueError, match='permutation symmetry'):
        scsf.CSFReferenceBackend(bad_ham)

def test_hci_iteration_variational_convergence():
    ham = load_hamiltonian(datafile('lih_sto3g.npz'))
    w = scsf.csf_wfn(ham.nbasis, 4, 0)
    w.add_csf('220000')
    op = scsf.csf_op(ham, w)
    energies, coeffs = davidson(op)
    for _ in range(10):
        old_energy = energies[0]
        added = scsf.add_csf_hci(ham, w, coeffs[0], eps=1e-05, backend=op.backend)
        if not added:
            break
        op.update(ham, w)
        energies, coeffs = davidson(op)
        assert energies[0] <= old_energy + 1e-13
    else:
        pytest.fail('tiny HCI search did not finish')
    full = scsf.csf_wfn(ham.nbasis, 4, 0)
    full.add_all_csfs()
    npt.assert_allclose(energies, davidson(scsf.csf_op(ham, full))[0], rtol=0, atol=1e-10)


@pytest.mark.parametrize('two_s,sector', [(0, 'singlet'), (2, 'triplet')])
def test_lih_independent_fci_roots(two_s, sector):
    ham = load_hamiltonian(datafile('lih_sto3g.npz'))
    w = scsf.csf_wfn(ham.nbasis, 4, two_s)
    w.add_all_csfs()
    energies, _ = davidson(scsf.csf_op(ham, w), n=3)
    with np.load(datafile('lih_sto3g.npz'), allow_pickle=False) as data:
        reference = data[sector]
    npt.assert_allclose(energies, reference, rtol=0, atol=2e-11)
