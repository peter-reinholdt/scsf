"""Selection variants preserve HCI support and publish valid variational spaces."""
import copy
import numpy as np
import numpy.testing as npt
import pytest
import scsf
from scsf.csf import _paths
from scsf.test.test_csf import random_hamiltonian


def initial(k=5,spin=0):
    full=scsf.csf_wfn(k,4,spin);full.add_all_csfs()
    base=scsf.csf_wfn(k,4,spin)
    for i in (0,7,12):base.add_csf(full[i])
    return base


@pytest.mark.parametrize('engine,spin,eps,workers,screening', [
    ('guga', 0, .2, 3, True), ('compiled', 2, .2, 1, False),
    ('guga', 2, 0., 1, False), ('compiled', 0, 1.5, 3, True),
])
def test_approximate_rules_contain_strict_hci_and_close_promoted_occupations(engine,spin,eps,workers,screening):
    ham=random_hamiltonian(5,seed=38)
    base=initial(spin=spin)
    strict=copy.copy(base)
    scsf.add_csf_hci(ham,strict,[.6,-.3,.1],eps=eps,
        backend=scsf.CSFSpinBackend(ham,coupling_engine=engine),screening=False)
    for rule in ('configuration','bound','transfer'):
        wfn=copy.copy(base)
        backend=scsf.CSFSpinBackend(ham,coupling_engine=engine,nthreads=workers,
                                   selection_engine='ordered',cache_bytes=0)
        _,stats=scsf.add_csf_hci(ham,wfn,[.6,-.3,.1],eps=eps,backend=backend,
                                selection_rule=rule,screening=screening,return_stats=True)
        assert set(strict)<=set(wfn)
        added=set(wfn)-set(base)
        for label in added:
            occ=tuple(2 if c=='2' else int(c!='0') for c in label)
            assert set(_paths(occ,spin))<=set(wfn)


def test_configuration_python_oracle_and_variational_hamiltonians():
    ham=random_hamiltonian(5,seed=23)
    base=initial()
    results={}
    for rule,implementation in (('hci','native'),('configuration','native'),('configuration','python'),
                                ('bound','native'),('transfer','native')):
        wfn=copy.copy(base)
        backend=(scsf.CSFReferenceBackend(ham) if implementation=='python' else
                 scsf.CSFSpinBackend(ham,coupling_engine='guga'))
        scsf.add_csf_hci(ham,wfn,[.6,-.3,.1],eps=.2,backend=backend,
                         implementation=implementation,selection_rule=rule)
        op=scsf.csf_op(ham,wfn,backend=scsf.CSFSpinBackend(ham,coupling_engine='guga'))
        energy=np.linalg.eigvalsh(op.to_scipy().toarray())[0]
        results[rule,implementation]=tuple(wfn),energy
    assert results['configuration','native'][0]==results['configuration','python'][0]
    for rule in ('configuration','bound','transfer'):
        assert results[rule,'native'][1]<=results['hci','native'][1]+1e-12


@pytest.mark.parametrize('rule,engine', [('hci','guga'), ('configuration','compiled'), ('bound','guga'), ('transfer','guga')])
def test_parent_norm_core_matches_explicitly_masked_generation(rule,engine,monkeypatch):
    ham=random_hamiltonian(5,seed=28)
    base=scsf.csf_wfn(5,4,0)
    for label in ('22000','2ud00','udud0','uudd0'):base.add_csf(label)
    coeffs=np.array([.8,.4,.3,-.2]); original=coeffs.copy()
    filtered=copy.copy(base); manual=copy.copy(base)
    backend=scsf.CSFSpinBackend(ham,coupling_engine=engine)
    from scsf import csf
    oracle_filter=csf._selection_parents
    def unexpected(*a,**kw):raise AssertionError('native filtering fell back to Python')
    monkeypatch.setattr(csf,'_selection_parents',unexpected)
    _,stats=scsf.add_csf_hci(ham,filtered,coeffs,eps=.2,backend=backend,
                           selection_rule=rule,parent_fraction=.8,return_stats=True)
    scsf.add_csf_hci(ham,manual,[.8,.4,0.,0.],eps=.2,backend=backend,selection_rule=rule)
    assert tuple(filtered)==tuple(manual)
    assert set(base)<=set(filtered)
    npt.assert_array_equal(coeffs,original)
    assert stats['parent_configurations']==3
    assert stats['retained_parent_configurations']==stats['retained_parent_csfs']==2
    assert stats['retained_parent_weight']==pytest.approx(.8/.93)
    if rule in ('hci','configuration'):
        monkeypatch.setattr(csf,'_selection_parents',oracle_filter)
        oracle=copy.copy(base)
        scsf.add_csf_hci(ham,oracle,coeffs,eps=.2,implementation='python',
                         selection_rule=rule,parent_fraction=.8)
        assert tuple(oracle)==tuple(filtered)
