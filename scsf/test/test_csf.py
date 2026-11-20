"""Spin sectors, basis I/O, and independent Hamiltonian references."""
import copy
import json
from math import comb

import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf.drivers.solvers import solve_ci
from scsf.test import datafile, load_hamiltonian
from .reference.oracle import determinant_keys, fock_reference, random_hamiltonian, transform_in_order


def solve(op, roots=1):
    energies, coeffs, converged = solve_ci(op.matmat, op.diagonal(), roots=roots,
                                          tol=1e-12, maxiter=300)
    assert converged
    return energies + op.ecore, coeffs.T


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
    op = scsf.csf_structured_op(ham, w)
    reference = u.T @ h @ u
    npt.assert_allclose(op.matmat(np.eye(len(w))), reference, rtol=0, atol=3e-14)
    npt.assert_allclose(h @ u, u @ reference, rtol=0, atol=2e-14)
    npt.assert_allclose(solve(op, roots=len(w))[0], np.linalg.eigvalsh(reference) + ham.ecore, rtol=0, atol=2e-14)

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

def test_molecular_determinant_and_converted_energy():
    ham = load_hamiltonian(datafile('lih_sto3g.npz'))
    w = scsf.csf_wfn(ham.nbasis, 4, 0)
    w.add_all_csfs()
    op = scsf.csf_structured_op(ham, w)
    energies, coeffs = solve(op)
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
    energies, _ = solve(scsf.csf_structured_op(ham, w), roots=3)
    npt.assert_allclose(energies, reference, rtol=0, atol=1e-08)
    npt.assert_allclose(energies - energies[0], np.array(reference) - reference[0], rtol=0, atol=1e-10)

@pytest.mark.parametrize('two_s,sector', [(0, 'singlet'), (2, 'triplet')])
def test_lih_independent_fci_roots(two_s, sector):
    ham = load_hamiltonian(datafile('lih_sto3g.npz'))
    w = scsf.csf_wfn(ham.nbasis, 4, two_s)
    w.add_all_csfs()
    energies, _ = solve(scsf.csf_structured_op(ham, w), roots=3)
    with np.load(datafile('lih_sto3g.npz'), allow_pickle=False) as data:
        reference = data[sector]
    npt.assert_allclose(energies, reference, rtol=0, atol=2e-11)
