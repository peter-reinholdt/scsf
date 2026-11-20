"""General spin-scalar one-electron products and sparse property selection."""
import copy
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from .reference.oracle import fock_operators


def model(k=4,n=4,s=0):
    w=scsf.csf_wfn(k,n,s);w.add_all_csfs()
    rng=np.random.default_rng(93)
    one=rng.normal(size=(k,k))+1j*rng.normal(size=(k,k))
    action=fock_operators(w)
    matrix=sum((one[p,q]*action((p,q)) for p in range(k) for q in range(k)),
               np.zeros((len(w),len(w)),complex))
    return w,one,matrix


@pytest.mark.parametrize('sector',[(0,0,0),(3,2,0),(4,4,0),(4,4,2),(5,3,1)])
@pytest.mark.parametrize('threads',[1,4,12])
def test_general_complex_product_against_fermion_actions(sector,threads):
    full,one,matrix=model(*sector)
    order=np.random.default_rng(17).permutation(len(full))[::2]
    w=scsf.csf_wfn(*sector)
    for j in order:w.add_csf(full[int(j)])
    expected=matrix[np.ix_(order,order)]
    rng=np.random.default_rng(34)
    x=rng.normal(size=(len(w),3))+1j*rng.normal(size=(len(w),3))
    before=tuple(w)
    for size in (len(w),len(w)//2,0):
        value=scsf.csf_one_electron_product(one,w,x[:size],nthreads=threads)
        npt.assert_allclose(value,expected[:,:size]@x[:size],rtol=0,atol=2e-14)
    npt.assert_allclose(scsf.csf_one_electron_product(one,w,x[:,0],nthreads=threads),
                        expected@x[:,0],rtol=0,atol=2e-14)
    npt.assert_allclose(scsf.csf_one_electron_product(np.eye(w.nbasis),w,x,nthreads=threads),
                        w.nelec*x,rtol=0,atol=2e-14)
    assert tuple(w)==before
    assert scsf.csf_one_electron_product(one,w,x[:,:0]).shape==(len(w),0)


@pytest.mark.parametrize('threads',[1,4,12])
def test_real_nonsymmetric_operator_and_adjoint(threads):
    w,one,matrix=model();one=one.real
    reference=fock_operators(w)
    expected=sum(one[p,q]*reference((p,q)) for p in range(4) for q in range(4))
    x=np.random.default_rng(9).normal(size=len(w))
    y=scsf.csf_one_electron_product(one,w,x,nthreads=threads)
    assert not np.iscomplexobj(y)
    npt.assert_allclose(y,expected@x,atol=1e-14)
    z=scsf.csf_one_electron_product(one.T,w,np.eye(len(w)),nthreads=threads)
    npt.assert_allclose(z,expected.T,atol=1e-14)
    assert not np.allclose(expected,expected.T)


@pytest.mark.parametrize('threads',[1,4,12])
@pytest.mark.parametrize('eps',[.2,2.,100.])
def test_one_electron_selection_against_independent_full_matrix(threads,eps):
    full,one,matrix=model()
    parent=[0,3,9]
    w=scsf.csf_wfn(4,4,0)
    for i in parent:w.add_csf(full[i])
    coeff=np.array([.3,-.4j,-.2]);saved=coeff.copy()
    matches=np.any(np.abs(matrix[:,parent]*coeff)>eps,axis=1)
    expected=sorted(full[i] for i in range(len(full)) if matches[i] and i not in parent)
    old=tuple(w)
    added,stats=scsf.add_csf_one_electron(one,w,coeff,eps,nthreads=threads,return_stats=True)
    assert tuple(w)==old+tuple(expected) and added==len(expected)==stats['added']
    rebuilt=scsf.csf_wfn(4,4,0)
    for label in w:rebuilt.add_csf(label)
    assert list(rebuilt._configs.items())==list(w._configs.items())
    npt.assert_array_equal(coeff,saved)


def test_diagonal_operator_does_not_grow_basis():
    w,one,_=model();base=scsf.csf_wfn(4,4,0);base.add_csf(w[0])
    assert scsf.add_csf_one_electron(np.eye(4),base,[1.],0.)==0
    assert tuple(base)==(w[0],)
    assert scsf.add_csf_one_electron(one,base,[1.],0.,nthreads=12)>0


def test_empty_inputs_validation_and_no_triplet_support():
    w=scsf.csf_wfn(3,2,0)
    assert scsf.csf_one_electron_product(np.eye(3),w,[]).shape==(0,)
    w.add_csf('200')
    for f in (scsf.csf_one_electron_product,scsf.add_csf_one_electron):
        with pytest.raises(NotImplementedError,match='triplet'):f(np.eye(3),w,[1.],triplet=True)
        with pytest.raises(ValueError):f(np.eye(2),w,[1.])
        with pytest.raises(ValueError):f(np.full((3,3),np.nan),w,[1.])
        with pytest.raises(ValueError):f(np.eye(3),w,[np.inf])
        with pytest.raises(ValueError):f(np.eye(3),w,[1.],nthreads=0)
    with pytest.raises(ValueError):scsf.csf_one_electron_product(np.eye(3),w,[1.,2.])
    with pytest.raises(ValueError):scsf.add_csf_one_electron(np.eye(3),w,[])


def test_long_orbital_intervals_and_no_hamiltonian_allocation(monkeypatch):
    w=scsf.csf_wfn(65,4,0)
    for a,b in (('2','0'),('u','d'),('0','2')):
        label=['0']*65;label[0]=a;label[31]='2';label[64]=b;w.add_csf(''.join(label))
    oracle=fock_operators(w)((64,0))
    one=np.zeros((65,65));one[64,0]=1.
    def forbidden(*args,**kwargs):raise AssertionError('one-body product allocated a Hamiltonian')
    monkeypatch.setattr(scsf,'hamiltonian',forbidden)
    npt.assert_allclose(scsf.csf_one_electron_product(one,w,np.eye(3),nthreads=12),oracle,atol=1e-14)


@pytest.mark.parametrize('threads',[1,12])
def test_many_open_shell_property_selection_uses_connected_walks(threads):
    from scsf.csf import _paths
    k=14;one=np.zeros((k,k));one[13,0]=1.
    w=scsf.csf_wfn(k,k,0);w.add_csf('ud'*7)
    occupation=[1]*k;occupation[0]=0;occupation[13]=2
    candidates=list(_paths(occupation,0))
    reference = copy.copy(w)
    for label in candidates:
        reference.add_csf(label)
    values = fock_operators(reference)((13,0))[1:,0]
    expected=sorted(label for label,value in zip(candidates,values) if abs(value)>.05)
    added,stats=scsf.add_csf_one_electron(one,w,[1.],.05,nthreads=threads,return_stats=True)
    assert tuple(w)[1:]==tuple(expected) and added==len(expected)
    assert stats['walk_graphs']>0
