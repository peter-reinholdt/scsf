"""Projected singlet response against dense selected-space resolvents."""
import copy
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf.drivers.response import resp, wrap_matvec
from scsf.drivers.solvers import solve_ci
from scsf.test.test_csf import random_hamiltonian

def solve(op):
    e, c, ok = solve_ci(lambda x: wrap_matvec(op, x), op.diagonal(), 1, tol=1e-12, maxiter=500)
    assert ok
    return (e, c)

def dense_response(op, c, b, shift=0):
    h = wrap_matvec(op, np.eye(len(c)))
    e = np.vdot(c, h @ c)
    q = np.eye(len(c)) - np.outer(c, c.conj())
    return np.linalg.solve(q @ (h - (e + shift) * np.eye(len(c))) @ q + np.outer(c, c.conj()), b)

@pytest.mark.parametrize('gamma,omega', [(0.0, 0.0), (0.03, 0.2)])
@pytest.mark.parametrize('complex_property', [False, True])
def test_general_property_response_against_dense(gamma, omega, complex_property):
    ham = random_hamiltonian(3, seed=21)
    w = scsf.csf_wfn(3, 2, 0)
    w.add_all_csfs()
    op = scsf.csf_op(ham, w, backend=scsf.CSFSpinBackend(ham, coupling_engine='guga'))
    _, c = solve(op)
    rng = np.random.default_rng(8)
    one = rng.normal(size=(3, 3))
    if complex_property:
        one = one + 1j * 1e-09 * rng.normal(size=(3, 3))
    parity = 1 if omega else 0
    r = resp(ham, w, op, c, {'A': one}, [('A', omega, parity)], gamma=gamma, ci_tol=1e-12, response_tol=1e-11, verbose=False)
    b = r[4]['A']
    x = r[3]['A', omega, parity]
    expected = dense_response(op, r[2][:, 0], b, parity * (omega + 1j * gamma))
    npt.assert_allclose(x, expected, rtol=0, atol=1e-11)
    factor = 1 if parity else 2
    npt.assert_allclose(r[5]['A', 'A', omega], factor * np.vdot(b, expected), atol=1e-11)
    assert abs(np.vdot(r[2][:, 0], x)) < 1e-12

@pytest.mark.parametrize('kind', ['csf'])
def test_multiple_property_selection_has_immutable_parents(kind):
    ham = scsf.hamiltonian(1.234, np.diag([-2.0, -1.0, 0.5]), np.zeros((3,) * 4))
    a = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    w = scsf.csf_wfn(3, 2, 0)
    w.add_csf('200')
    op = scsf.csf_op(ham, w, backend=scsf.CSFSpinBackend(ham, coupling_engine='guga', nthreads=4))
    _, c = solve(op)
    old = c.copy()
    stored = op.data().copy()
    r = resp(ham, w, op, c, {'A': a, 'B': b}, [('A', 0.0, 0), ('B', 0.0, 0)], eps_mu=0.0, eps_resp=0.0, overwrite=False, response_growth_tol=0.0, verbose=False)
    assert len(w) == 1 and len(r[0]) == 2
    npt.assert_array_equal(c, old)
    npt.assert_array_equal(op.data(), stored)
    assert r[5]['A', 'A', 0.0] == pytest.approx(4.0, abs=1e-12)
    assert abs(r[5]['B', 'B', 0.0]) < 1e-15

@pytest.mark.parametrize('eps_mu,eps_resp', [(None, None), (0.1, None), (None, 0.1), (0.1, 0.1)])
def test_sparse_csf_growth_and_dense_final_response(eps_mu, eps_resp):
    ham = random_hamiltonian(4, seed=51)
    full = scsf.csf_wfn(4, 4, 0)
    full.add_all_csfs()
    w = scsf.csf_wfn(4, 4, 0)
    for label in full[:3]:
        w.add_csf(label)
    op = scsf.csf_op(ham, w, backend=scsf.CSFSpinBackend(ham, coupling_engine='guga', nthreads=4))
    _, c = solve(op)
    one = np.random.default_rng(19).normal(size=(4, 4))
    one = (one + one.T) / 2
    r = resp(ham, w, op, c, {'A': one}, [('A', 0.0, 0)], eps_mu=eps_mu, eps_resp=eps_resp, response_growth_tol=0.0, response_tol=1e-10, ci_tol=1e-12, verbose=False)
    assert len(r[0]) >= 3
    if eps_mu is not None or eps_resp is not None:
        assert len(r[0]) > 3
    expected = dense_response(r[1], r[2][:, 0], r[4]['A'])
    npt.assert_allclose(r[3]['A', 0.0, 0], expected, atol=1e-10, rtol=0)

def test_csf_triplet_rejected_and_zero_property():
    ham = random_hamiltonian(3)
    w = scsf.csf_wfn(3, 2, 0)
    w.add_all_csfs()
    op = scsf.csf_op(ham, w)
    _, c = solve(op)
    with pytest.raises(NotImplementedError, match='triplet'):
        resp(ham, w, op, c, {}, [], triplet=True)
    r = resp(ham, w, op, c, {'Z': np.zeros((3, 3))}, [('Z', 0.0, 0)], verbose=False)
    npt.assert_array_equal(r[3]['Z', 0.0, 0], np.zeros(len(w)))
    assert r[-1]['Z', 'Z', 0.0] == 0
