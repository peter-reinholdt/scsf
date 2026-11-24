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


def test_davidson_phase_stats_preserve_solution_and_count_work():
    rng = np.random.default_rng(781)
    matrix = rng.normal(size=(25,25)); matrix = (matrix + matrix.T)/2
    stats, calls = {}, []
    def product(x):
        calls.append(x.shape[1])
        return matrix @ x
    measured = solve_ci(product, np.diag(matrix), 2, tol=1e-9, maxiter=250,
                        max_subspace=12, stats=stats)
    control = solve_ci(lambda x:matrix @ x, np.diag(matrix), 2, tol=1e-9,
                       maxiter=250, max_subspace=12)
    npt.assert_allclose(measured[0], control[0], atol=1e-12)
    assert measured[2] == control[2]
    energies, vectors, _ = measured
    npt.assert_allclose(stats['residual_norms'], np.linalg.norm(matrix @ vectors - vectors * energies, axis=0),
                        atol=1e-13, rtol=0)
    assert stats['product_calls'] == len(calls)
    assert stats['product_columns'] == sum(calls)
    assert stats['call_widths'] == calls
    assert stats['iterations'] > 1 and stats['restarts'] > 0
    phases = [stats[key] for key in ('product_ns', 'orthogonalization_ns', 'ritz_ns',
                                    'restart_ns', 'basis_append_ns', 'other_ns')]
    assert all(v >= 0 for v in phases)
    assert sum(phases) == stats['total_ns']
