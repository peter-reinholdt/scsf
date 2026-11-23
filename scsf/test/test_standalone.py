"""Contracts introduced by the standalone extraction."""
import json
import numpy as np
import numpy.testing as npt
import pytest
import scsf


def test_integral_container_owns_contiguous_array_storage():
    h = np.ones((3,3)).T
    g = np.zeros((3,)*4)
    ham = scsf.hamiltonian(1.25, h, g)
    h[:] = -1.
    g[:] = 2.
    assert ham.nbasis == 3 and ham.ecore == 1.25
    assert ham.one_mo.flags.c_contiguous and ham.two_mo.flags.c_contiguous
    npt.assert_array_equal(ham.one_mo, 1.)
    npt.assert_array_equal(ham.two_mo, 0.)


@pytest.mark.parametrize('core,one,two,error', [
    (0., np.zeros(2), np.zeros((2,)*4), ValueError),
    (0., np.zeros((2,2)), np.zeros((2,2)), ValueError),
    (np.nan, np.zeros((2,2)), np.zeros((2,)*4), ValueError),
    (0., np.full((2,2), np.inf), np.zeros((2,)*4), ValueError),
    (0., np.zeros((2,2),complex), np.zeros((2,)*4), TypeError),
])
def test_invalid_integral_snapshots(core,one,two,error):
    with pytest.raises(error):
        scsf.hamiltonian(core,one,two)


def test_basis_file_round_trip(tmp_path):
    w = scsf.csf_wfn(2,2,0)
    w.add_all_csfs()
    path = tmp_path/'basis.json'
    w.to_file(path)
    payload = json.loads(path.read_text())
    assert set(payload) == {'nbasis','nelec','two_s','labels'}
    restored = scsf.csf_wfn.from_file(path)
    assert list(restored) == list(w)


def test_array_snapshot_driver_requires_electron_count(tmp_path):
    from scsf.drivers.csf_driver import build_parser, integrals
    snapshot = tmp_path/'integrals.npz'
    np.savez(snapshot, ecore=0., one=np.eye(2), two=np.zeros((2,)*4))
    parser = build_parser()
    args = parser.parse_args(['--integrals', str(snapshot)])
    with pytest.raises(SystemExit):
        integrals(args,parser)
    args.nelec = 2
    ham, nelec = integrals(args,parser)
    assert ham.nbasis == nelec == 2


def test_basis_files_have_one_schema(tmp_path):
    path=tmp_path/'basis.json'
    w=scsf.csf_wfn(2,2,0);w.add_csf('20');w.to_file(path)
    payload=json.loads(path.read_text());payload['obsolete_metadata']=True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='basis file'):
        scsf.csf_wfn.from_file(path)


def test_dense_integrals_require_physical_symmetry():
    with pytest.raises(ValueError, match='symmetric'):
        scsf.Hamiltonian(0., [[0., 1.], [0., 0.]], np.zeros((2,)*4))
    eri = np.zeros((2,)*4)
    eri[0, 0, 1, 0] = 1.
    with pytest.raises(ValueError, match='permutation symmetry'):
        scsf.Hamiltonian(0., np.eye(2), eri)
