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

"""Native SCI connectivity, bounded selection, and incremental Hamiltonians."""

import copy
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

import scsf
from scsf.csf import connected_configurations
from scsf.test.test_csf import random_hamiltonian


@pytest.mark.parametrize("k", [0,3])
def test_native_configuration_endpoints_exhaustive(k):
    backend = scsf.CSFSpinBackend(random_hamiltonian(k))
    for config in product(range(3),repeat=k):
        assert backend.connected_configurations(config) == list(connected_configurations(config))


@pytest.mark.parametrize("n,s", [(4,0),(4,2),(3,1)])
@pytest.mark.parametrize("eps", [0.,.5,1e8])
def test_native_selection_exact_labels(n,s,eps):
    ham = random_hamiltonian(4,seed=218)
    full = scsf.csf_wfn(4,n,s)
    full.add_all_csfs()
    w = scsf.csf_wfn(4,n,s)
    for i in sorted({0,min(3,len(full)-1),len(full)-1}):
        w.add_csf(full[i])
    if s == 0 and n == 4:
        w.add_csf("uudd")  # retain only one path of this configuration
    coeffs = np.linspace(.7,-.2,len(w))
    coeffs[0] = 0.
    native = scsf.CSFSpinBackend(ham)
    answers = []
    for implementation,screening in (("native",True),("native",False),("python",False)):
        trial = copy.copy(w)
        count,stats = scsf.add_csf_hci(
            ham,trial,coeffs,eps=eps,backend=native,implementation=implementation,
            screening=screening,return_stats=True)
        assert count == stats["added"] == len(trial)-len(w)
        assert tuple(trial)[len(w):] == tuple(sorted(set(trial)-set(w)))
        answers.append(tuple(trial))
    assert answers[0] == answers[1] == answers[2]
    if eps:
        reference = copy.copy(w)
        scsf.add_csf_hci(ham,reference,coeffs,eps=eps,backend=scsf.CSFReferenceBackend(ham))
        assert answers[0] == tuple(reference)


def test_bound_covers_every_target_path_and_screening_is_effective():
    ham = random_hamiltonian(4)
    backend = scsf.CSFSpinBackend(ham)
    w = scsf.csf_wfn(4,4,0)
    w.add_all_csfs()
    matrix = scsf.CSFReferenceBackend(ham).block(w,w,0)
    for a,ia in w._configs.items():
        for b,ib in w._configs.items():
            bound = backend.configuration_bound(a,b)
            assert np.max(np.abs(matrix[np.ix_(ia,ib)])) <= bound

    h = np.zeros((6,6))
    h[0,1] = h[1,0] = .1
    ham = scsf.hamiltonian(0.,h,np.zeros((6,)*4))
    backend = scsf.CSFSpinBackend(ham)
    base = scsf.csf_wfn(6,2,0)
    base.add_csf("200000")
    a,b = copy.copy(base),copy.copy(base)
    _,screened = scsf.add_csf_hci(ham,a,[1.],eps=.01,backend=backend,return_stats=True)
    _,plain = scsf.add_csf_hci(ham,b,[1.],eps=.01,backend=backend,
                               screening=False,return_stats=True)
    assert tuple(a) == tuple(b) and len(a) == 2
    assert screened["screened_pairs"] > 0
    assert screened["matrix_elements"] < plain["matrix_elements"]
    assert screened["candidates"] < plain["candidates"]
    assert backend.last_selection_stats == plain


def test_screening_precedes_spin_path_enumeration_and_allocation():
    k = 10
    ham = scsf.hamiltonian(0.,np.diag(np.arange(1.,k+1)),np.zeros((k,)*4))
    backend = scsf.CSFSpinBackend(ham,max_tensor_entries=1)
    assert backend.configuration_bound([1]*k,[1]*k) >= sum(range(1,k+1))
    w = scsf.csf_wfn(k,k,0)
    w.add_csf("u"*5+"d"*5)
    count,stats = scsf.add_csf_hci(ham,w,[1.],eps=0,backend=backend,
                                  max_candidates=0,max_csfs=1,return_stats=True)
    assert count == stats["matrix_elements"] == stats["candidates"] == 0
    assert stats["screened_pairs"] == stats["configuration_pairs"]
    with pytest.raises(MemoryError):
        scsf.add_csf_hci(ham,w,[1.],eps=0,backend=backend,
                         screening=False,max_candidates=0,max_csfs=1)
    assert tuple(w) == ("u"*5+"d"*5,)


@pytest.mark.parametrize("eps,added", [(.125,0),(.125-1e-15,1),(.125+1e-15,0),
                                     (0.,1),(1e-13,1)])
def test_strict_threshold_and_no_matrix_truncation(eps,added):
    ham = scsf.hamiltonian(0.,np.array([[0.,.125],[.125,0.]]),np.zeros((2,)*4))
    backend = scsf.CSFSpinBackend(ham)
    w = scsf.csf_wfn(2,1,1)
    w.add_csf("u0")
    assert scsf.add_csf_hci(ham,w,[1.],eps=eps,backend=backend) == added
    ham = scsf.hamiltonian(1.,np.array([[0.,1e-13],[1e-13,0.]]),np.zeros((2,)*4))
    backend = scsf.CSFSpinBackend(ham)
    w = scsf.csf_wfn(2,1,1)
    w.add_csf("u0"); w.add_csf("0u")
    op = scsf.csf_op(ham,w,backend=backend,matrix_cutoff=0.)
    assert op.get_element(0,1) == 1e-13
    assert op.size == 3 and np.all(op.diagonal() == 0.)


@pytest.mark.parametrize("k,n,s", [(0,0,0),(4,4,0),(4,4,2),(4,3,1),(4,6,0)])
def test_native_csr_full_rectangular_and_incremental(k,n,s):
    ham = random_hamiltonian(k)
    backend = scsf.CSFSpinBackend(ham)
    full = scsf.csf_wfn(k,n,s)
    full.add_all_csfs()
    reference = scsf.CSFReferenceBackend(ham).block(full,full,s)
    for symmetric in (True,False):
        op = scsf.csf_op(ham,full,backend=backend,symmetric=symmetric)
        npt.assert_allclose(op.to_scipy().toarray(),reference,rtol=0,atol=3e-14)
        assert backend.last_matrix_stats["rows"] == len(full)
        for row in range(len(full)):
            columns = op.indices()[op.indptr()[row]:op.indptr()[row+1]]
            assert np.all(np.diff(columns) > 0)
    for nrow,ncol in ((0,len(full)),(min(3,len(full)),len(full)),(len(full),0)):
        rect = scsf.csf_op(ham,full,nrow=nrow,ncol=ncol,symmetric=False,backend=backend)
        npt.assert_allclose(rect.to_scipy().toarray(),reference[:nrow,:ncol],rtol=0,atol=3e-14)
    selected = scsf.csf_wfn(k,n,s)
    selected.add_csf(full[-1])
    triangular = scsf.csf_op(ham,selected,backend=backend)
    general = scsf.csf_op(ham,selected,backend=backend,symmetric=False)
    before = triangular.data()
    for label in full:
        selected.add_csf(label)
    triangular.update(ham,selected)
    if len(full) > 1:
        assert backend.last_matrix_stats["rows"] == len(full)-1
    npt.assert_array_equal(triangular.data()[:len(before)],before)
    general.update(ham,selected)
    rebuilt = scsf.csf_op(ham,selected,backend=backend)
    npt.assert_array_equal(triangular.to_scipy().toarray(),rebuilt.to_scipy().toarray())
    npt.assert_allclose(general.to_scipy().toarray(),rebuilt.to_scipy().toarray(),atol=3e-14)


def test_native_budget_failures_leave_state_unchanged():
    ham = random_hamiltonian(4)
    w = scsf.csf_wfn(4,4,0)
    w.add_csf("uudd")
    backend = scsf.CSFSpinBackend(ham)
    for kwargs in ({"max_candidates":0},{"max_csfs":1},{"max_configurations":1}):
        with pytest.raises(MemoryError):
            scsf.add_csf_hci(ham,w,[1.],eps=0,backend=backend,screening=False,**kwargs)
        assert tuple(w) == ("uudd",)
        assert backend.last_selection_stats == {}
    for limit in ("max_tensor_entries","max_bond_elements","max_connection_terms"):
        guarded = scsf.CSFSpinBackend(ham,**{limit:1})
        with pytest.raises(MemoryError):
            scsf.add_csf_hci(ham,w,[1.],eps=0,backend=guarded,screening=False)
        assert tuple(w) == ("uudd",)
    op = scsf.csf_op(ham,w,backend=backend)
    data,shape = op.data(),op.shape
    previous_stats = dict(backend.last_matrix_stats)
    w.add_csf("udud")
    op.max_matrix_entries = len(data)
    with pytest.raises(MemoryError):
        op.update(ham,w)
    assert op.shape == shape and backend.last_matrix_stats == previous_stats
    npt.assert_array_equal(op.data(),data)
    op.max_matrix_entries = 1000
    op.update(ham,w)
    assert op.shape == (2,2)


def test_screening_evaluates_floating_point_boundary_band():
    g = np.zeros((2,)*4)
    for index in ((0,1,0,1),(1,0,0,1),(0,1,1,0),(1,0,1,0)):
        g[index] = 1.
    ham = scsf.hamiltonian(0.,np.zeros((2,2)),g.transpose(0,2,1,3))
    backend = scsf.CSFSpinBackend(ham,coupling_engine='compiled')
    for eps,expected in ((1.,0),(1.-1e-14,1),(1.+1e-14,0)):
        w = scsf.csf_wfn(2,2,0)
        w.add_csf("20")
        count,stats = scsf.add_csf_hci(ham,w,[1.],eps=eps,backend=backend,return_stats=True)
        assert count == expected
        assert stats["matrix_elements"] > 0


def test_native_sci_input_validation():
    from scsf import _csf

    ham = random_hamiltonian(4)
    backend = scsf.CSFSpinBackend(ham)
    w = scsf.csf_wfn(4,4,0)
    w.add_csf("2200")
    with pytest.raises(ValueError,match="implementation"):
        scsf.add_csf_hci(ham,w,[1.],backend=backend,implementation="invalid")
    with pytest.raises(TypeError,match="CSFSpinBackend"):
        scsf.add_csf_hci(ham,w,[1.],implementation="native",backend=scsf.CSFReferenceBackend(ham))
    with pytest.raises(ValueError):
        backend.connected_configurations([3,1,0,0])
    with pytest.raises(ValueError):
        backend.configuration_bound([2,2,0,0],[1,1,0,0])
    with pytest.raises(ValueError):
        _csf.select_hci(ham.one_mo,ham.two_mo,list(w),0,np.ones(2),.1)
    with pytest.raises(ValueError,match="duplicate"):
        _csf.select_hci(ham.one_mo,ham.two_mo,["2200"]*2,0,np.ones(2),.1)
    with pytest.raises(ValueError):
        _csf.build_csr(ham.one_mo,ham.two_mo,list(w),0,2,1)
    with pytest.raises(ValueError):
        _csf.build_csr(ham.one_mo,ham.two_mo,list(w),0,1,1,False,1)
