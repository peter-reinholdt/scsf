# This file is part of SCSF.
#
# SCSF is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# SCSF is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with SCSF. If not, see <http://www.gnu.org/licenses/>.

"""Davidson regression tests independent of the native CSF extension."""

import numpy as np
import numpy.testing as npt
import pytest

from scsf.drivers.solvers import solve_ci
from scsf.drivers.solvers import davidson_response


def test_response_preserves_small_imaginary_components():
    a=np.array([[2.,.2,.1],[.2,3.,.3],[.1,.3,4.]])
    b=np.array([1.,.4,.7])+1j*1e-9*np.array([.3,-.2,.5])
    x=davidson_response(a.__matmul__,b,np.diag(a),tol=1e-12)
    npt.assert_allclose(x,np.linalg.solve(a,b),rtol=0,atol=1e-12)
    assert np.linalg.norm(a@x-b)<1e-12


@pytest.mark.parametrize("dim,roots", [(1,1),(3,3),(12,1),(12,3)])
def test_davidson_exact_roots(dim,roots):
    rng = np.random.default_rng(171)
    matrix = rng.normal(size=(dim,dim))
    matrix = (matrix+matrix.T)/2
    energies, vectors, converged = solve_ci(matrix.__matmul__,np.diag(matrix),roots,
                                            tol=1e-11,max_subspace=dim)
    assert converged
    npt.assert_allclose(energies,np.linalg.eigvalsh(matrix)[:roots],rtol=0,atol=1e-12)
    npt.assert_allclose(matrix@vectors,vectors*energies,rtol=0,atol=1e-11)
    npt.assert_allclose(vectors.T@vectors,np.eye(roots),rtol=0,atol=1e-12)


def test_restarts_rank_deficient_guesses_and_nonconvergence():
    rng = np.random.default_rng(32)
    matrix = rng.normal(scale=.05,size=(60,60))
    matrix = (matrix+matrix.T)/2+np.diag(np.arange(60.))
    c0 = np.zeros((60,4))
    c0[0,:2] = 1.
    energies,vectors,converged = solve_ci(matrix.__matmul__,np.diag(matrix),3,c0=c0,
                                          tol=1e-10,maxiter=300,max_subspace=8)
    assert converged
    npt.assert_allclose(energies,np.linalg.eigvalsh(matrix)[:3],rtol=0,atol=1e-11)
    assert vectors.shape == (60,3)
    _,_,converged = solve_ci(matrix.__matmul__,np.diag(matrix),3,maxiter=1,tol=1e-14)
    assert not converged


def test_reuse_guesses_after_basis_expansion_and_diagonal_stagnation():
    matrix = np.diag([2.,3.,-1.])
    c0 = np.array([[1.],[0.],[0.]])
    energies,_,converged = solve_ci(matrix.__matmul__,np.diag(matrix),1,c0=c0)
    assert converged
    npt.assert_allclose(energies,[-1.])
    matrix = np.ones((6,6))-np.eye(6)
    energies,_,converged = solve_ci(matrix.__matmul__,np.diag(matrix),3,tol=1e-10)
    assert converged
    npt.assert_allclose(energies,[-1.]*3,atol=1e-11)


def test_tracking_and_hvp_validation():
    matrix = np.diag([0.,1.,2.,3.])
    e,x,ok = solve_ci(matrix.__matmul__,np.diag(matrix),1,c0=np.eye(4),
                      reference_state=np.array([0.,0.,1.]))
    assert ok and x.shape == (4,3)
    npt.assert_allclose(e,[0.,1.,2.])
    with pytest.raises(ValueError,match="hvp"):
        solve_ci(lambda x: np.zeros(4),np.zeros(4),1)
    with pytest.raises(ValueError,match="c0"):
        solve_ci(matrix.__matmul__,np.diag(matrix),1,c0=np.ones(2))


@pytest.mark.parametrize('roots,threshold,decay,source', [
    (1, .01, 0., 'orthogonal'), (3, .05, .5, 'orthogonal'),
    (1, .05, .5, 'preconditioned'), (3, .01, 0., 'preconditioned'),
])
def test_sparse_directions_keep_exact_images_through_restarts(roots, threshold, decay, source):
    rng = np.random.default_rng(32)
    matrix = rng.normal(scale=.05, size=(60, 60))
    matrix = (matrix+matrix.T)/2 + np.diag(np.arange(60.))
    calls = []
    def apply(x):
        calls.append(x.copy())
        return matrix @ x
    c0 = np.zeros((60, 4))
    c0[0, :2] = 1.
    e, c, ok = solve_ci(apply, np.diag(matrix), roots, c0=c0,
        tol=1e-10, maxiter=300, max_subspace=8, trial_vector_tol=threshold,
        trial_vector_decay=decay, trial_vector_source=source)
    assert ok
    npt.assert_allclose(e, np.linalg.eigvalsh(matrix)[:roots], rtol=0, atol=1e-11)
    assert np.max(np.linalg.norm(matrix @ c-c*e, axis=0)) < 1e-10
    npt.assert_allclose(c.T @ c, np.eye(roots), rtol=0, atol=1e-12)
    # Apart from the initial guess, the native operator receives unit-norm
    # sparse columns; the Ritz vector itself need not have this sparsity.
    assert any(np.count_nonzero(x) < x.size for x in calls[1:])
    if threshold >= .05 and decay == 0:
        assert any(np.count_nonzero(x) < x.size//2 for x in calls[1:])
    for x in calls[1:]:
        npt.assert_allclose(np.linalg.norm(x, axis=0), 1., rtol=0, atol=1e-12)


def test_sparse_direction_dependence_falls_back_without_false_convergence():
    # A very aggressive cutoff retains only the same coordinate as the seed.
    # It must recover the original dense correction and its consistent image.
    matrix = np.diag(np.arange(8.))
    matrix[0, 1:] = matrix[1:, 0] = .01
    c0 = np.ones(8)
    e, c, ok = solve_ci(matrix.__matmul__, np.diag(matrix), 1, c0=c0,
        tol=1e-11, maxiter=200, max_subspace=8, trial_vector_tol=.99)
    assert ok
    npt.assert_allclose(e, np.linalg.eigvalsh(matrix)[:1], rtol=0, atol=1e-12)
    assert np.linalg.norm(matrix @ c-c*e) < 1e-11


def test_sparse_directions_zero_corrections_and_tracking():
    matrix = np.diag([0., 1., 2., 3.])
    e, c, ok = solve_ci(matrix.__matmul__, np.diag(matrix), 1, c0=np.eye(4),
        reference_state=np.array([0., 0., 1.]), trial_vector_tol=.1)
    assert ok
    npt.assert_allclose(e, [0., 1., 2.])
    matrix = np.ones((6, 6))-np.eye(6)
    e, c, ok = solve_ci(matrix.__matmul__, np.diag(matrix), 3,
        tol=1e-11, trial_vector_tol=.1)
    assert ok
    npt.assert_allclose(e, [-1.]*3, atol=1e-12)
    assert np.linalg.norm(matrix @ c-c*e) < 1e-11


@pytest.mark.parametrize('threshold', [-1., 1., np.nan, np.inf])
def test_invalid_trial_vector_threshold(threshold):
    with pytest.raises(ValueError, match='trial_vector_tol'):
        solve_ci(np.eye(3).__matmul__, np.arange(3.), 1, trial_vector_tol=threshold)


@pytest.mark.parametrize('decay', [-1., np.nan, np.inf])
def test_invalid_trial_vector_decay(decay):
    with pytest.raises(ValueError, match='trial_vector_decay'):
        solve_ci(np.eye(3).__matmul__, np.arange(3.), 1, trial_vector_decay=decay)


def test_trial_vector_source_validation_and_threshold_equality():
    with pytest.raises(ValueError, match='trial_vector_source'):
        solve_ci(np.eye(3).__matmul__, np.arange(3.), 1, trial_vector_source='bad')
    matrix = np.array([[0., .6, .8], [.6, 1., 0.], [.8, 0., 1.]])
    for threshold, support in [(.6, 2), (np.nextafter(.6, 1.), 1)]:
        calls = []
        def apply(x):
            calls.append(x.copy())
            return matrix @ x
        solve_ci(apply, np.diag(matrix), 1, maxiter=2, trial_vector_tol=threshold)
        assert np.count_nonzero(calls[1]) == support


def test_trial_vector_decay_restores_fuller_inputs_and_resets_per_solve():
    rng = np.random.default_rng(32)
    matrix = rng.normal(scale=.05, size=(60, 60))
    matrix = (matrix+matrix.T)/2 + np.diag(np.arange(60.))
    previous = None
    for _ in range(2):
        calls = []
        def apply(x):
            calls.append(x.copy())
            return matrix @ x
        e, c, ok = solve_ci(apply, np.diag(matrix), 1, tol=1e-11,
            trial_vector_tol=.2, trial_vector_decay=.5, maxiter=200)
        assert ok and np.linalg.norm(matrix @ c-c*e) < 1e-11
        assert np.count_nonzero(calls[-1]) > np.count_nonzero(calls[1])
        if previous is not None:
            for a, b in zip(previous, calls):
                npt.assert_array_equal(a, b)
        previous = calls


def test_zero_trial_vector_threshold_preserves_reference_calls():
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(15, 15))
    matrix = (matrix+matrix.T)/2
    recorded = []
    def apply(x):
        recorded.append(x.copy())
        return matrix @ x
    first = solve_ci(apply, np.diag(matrix), 2, max_subspace=15)
    reference = recorded.copy()
    recorded.clear()
    second = solve_ci(apply, np.diag(matrix), 2, max_subspace=15, trial_vector_tol=0.)
    assert first[2] and second[2] and len(reference) == len(recorded)
    for a, b in zip(reference, recorded):
        npt.assert_array_equal(a, b)
    npt.assert_array_equal(first[0], second[0])
    npt.assert_array_equal(first[1], second[1])
