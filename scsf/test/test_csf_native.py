from scsf.test import load_hamiltonian
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

"""Native spin contraction versus the independent determinant reference."""

import copy
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

import scsf
from scsf.drivers.csf_driver import solve_variational
from scsf.test import datafile
from scsf.test.test_csf import random_hamiltonian


@pytest.mark.parametrize("k,n,s", [
    (k,n,s) for k in range(5) for n in range(2*k+1)
    for s in range(n % 2,min(n,2*k-n)+1,2)
])
def test_complete_native_blocks(k,n,s):
    ham = random_hamiltonian(k)
    w = scsf.csf_wfn(k,n,s)
    w.add_all_csfs()
    native, reference = scsf.CSFSpinBackend(ham), scsf.CSFReferenceBackend(ham)
    matrix = native.block(w,w,s)
    expected = reference.block(w,w,s)
    npt.assert_allclose(matrix,expected,rtol=0,atol=3e-14)
    npt.assert_allclose(matrix,matrix.T,rtol=0,atol=2e-14)
    # Individual element calls use the same labels/phases as block evaluation.
    for i,j in ((0,len(w)-1),(len(w)-1,0)):
        npt.assert_allclose(native.element(w[i],w[j],s),expected[i,j],rtol=0,atol=3e-14)


@pytest.mark.parametrize("s", [0,2,4,6])
def test_six_open_shells_with_double_occupations(s):
    ham = random_hamiltonian(8,seed=73)
    w = scsf.csf_wfn(8,8,s)
    for config in ([2,1,1,1,1,1,1,0], [1,2,1,1,1,1,0,1]):
        w.add_configuration(config)
    native, reference = scsf.CSFSpinBackend(ham),scsf.CSFReferenceBackend(ham)
    npt.assert_allclose(native.block(w,w,s),reference.block(w,w,s),rtol=0,atol=1e-13)


@pytest.mark.parametrize("s", [0,2])
def test_native_operator_updates_selection_and_dalton(s):
    ham = load_hamiltonian(datafile("h4_dalton_sto3g.npz"))
    full = scsf.csf_wfn(4,4,s)
    full.add_all_csfs()
    native, reference = scsf.CSFSpinBackend(ham), scsf.CSFReferenceBackend(ham)
    matrix = reference.block(full,full,s)
    op = scsf.csf_op(ham,full,backend=native,matrix_cutoff=0.)
    energies, vectors = solve_variational(op,3)
    npt.assert_allclose(energies,np.linalg.eigvalsh(matrix)[:3]+ham.ecore,rtol=0,atol=2e-13)
    for e,c in zip(energies,vectors.T):
        npt.assert_allclose(op(c),(e-ham.ecore)*c,rtol=0,atol=1e-10)
    rect = scsf.csf_op(ham,full,nrow=3,ncol=7,symmetric=False,backend=native)
    npt.assert_allclose(rect.to_scipy().toarray(),matrix[:3,:7],rtol=0,atol=3e-14)
    npt.assert_allclose(op.matmat(vectors),matrix@vectors,rtol=0,atol=3e-14)
    assert op.matmat(np.empty((len(full),0))).shape == (len(full),0)
    w = scsf.csf_wfn(4,4,s)
    w.add_csf("uudd" if s == 0 else "uuud")
    w.add_csf("2200" if s == 0 else "2uu0")
    incremental = scsf.csf_op(ham,w,backend=native)
    initial = tuple(w)
    for eps in (.5,.1,.01):
        a,b = copy.copy(w),copy.copy(w)
        coeffs = np.linspace(.3,-.7,len(w))
        assert scsf.add_csf_hci(ham,a,coeffs,eps=eps,backend=native) == scsf.add_csf_hci(
            ham,b,coeffs,eps=eps,backend=reference)
        assert tuple(a) == tuple(b)
    w.add_configuration([1]*4)
    assert tuple(w)[:len(initial)] == initial
    incremental.update(ham,w)
    rebuilt = scsf.csf_op(ham,w,backend=native)
    npt.assert_array_equal(incremental.to_scipy().toarray(),rebuilt.to_scipy().toarray())


def exchange_hamiltonian(k):
    # (pq|qp)=1, p!=q, with all eight real-integral permutations. In the
    # all-singly-occupied block H=-sum_{p<q} P_spin(p,q).
    g = np.zeros((k,)*4)
    for p,q in product(range(k),repeat=2):
        if p != q:
            for a,b,c,d in ((p,q,q,p),(q,p,q,p),(p,q,p,q),(q,p,p,q)):
                g[a,b,c,d] = 1.
    return scsf.hamiltonian(0.,np.zeros((k,k)),g.transpose(0,2,1,3))


@pytest.mark.parametrize("two_s", [0,2,4])
def test_many_open_shells_without_determinants(monkeypatch,two_s):
    import scsf.csf as csf

    k = 24
    ham = exchange_hamiltonian(k)
    w = scsf.csf_wfn(k,k,two_s)
    # Two distinct valid paths, each with a determinant support above the
    # default reference limit. Analytic exchange sum depends only on total S.
    pairs = (k-two_s)//2
    w.add_csf("u"*(pairs+two_s)+"d"*pairs)
    w.add_csf("ud"*pairs+"u"*two_s)
    with pytest.raises(MemoryError):
        scsf.CSFReferenceBackend(ham).element(w[0],w[0],two_s)

    def forbidden_expansion(*args,**kwargs):
        pytest.fail("native coupling attempted a determinant expansion")

    monkeypatch.setattr(csf,"_expand",forbidden_expansion)
    native = scsf.CSFSpinBackend(ham)
    matrix = native.block(w,w,two_s)
    spin = two_s/2
    expected = -(spin*(spin+1)+k*(k-4)/4)
    npt.assert_allclose(matrix,np.eye(len(w))*expected,rtol=0,atol=5e-12)


def test_native_guards_and_raw_validation():
    from scsf import _csf

    ham = random_hamiltonian()
    for kwargs in ({"max_block_elements":0},{"max_bond_elements":1},
                   {"max_tensor_entries":1}):
        backend = scsf.CSFSpinBackend(ham,**kwargs)
        with pytest.raises(MemoryError):
            backend.element("uudd","uudd",0)
    backend = scsf.CSFSpinBackend(ham)
    for a,b in (("duud","uudd"),("uudd","uuuu"),("uudd","ud00")):
        with pytest.raises(ValueError):
            backend.element(a,b,0)
    with pytest.raises(ValueError):
        _csf.hamiltonian_block(ham.one_mo,ham.two_mo,["duud"],["uudd"],0)
    with pytest.raises(ValueError,match="max_tensor_entries"):
        _csf.hamiltonian_block(ham.one_mo,ham.two_mo,["uudd"],["uudd"],0,100,1)
