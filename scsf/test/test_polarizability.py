"""Molecular singlet polarizability from fresh CSF spaces."""
import numpy as np
import numpy.testing as npt
import pytest
import subprocess
import sys
from pathlib import Path
from scsf.drivers import polarizability


@pytest.fixture
def molecule(tmp_path):
    pytest.importorskip('pyscf')
    path=tmp_path/'h4.xyz'
    path.write_text('4\nLinear H4\nH 0 0 -1.2\nH 0 0 -.4\nH 0 0 .4\nH 0 0 1.2\n')
    return str(path)


def options(molecule):
    return ['--xyz',molecule,'--basis','sto-3g','--nthreads','2',
            '--response-tol','1e-10','--davidson-tol','1e-12']


def test_full_tensor(molecule,capsys):
    r=polarizability.main(options(molecule)+['--full'])
    assert 'CSF singlet response' in capsys.readouterr().out
    assert r[-1][('Z','Z',0.)]==pytest.approx(11.107698996,abs=2e-8)
    assert abs(r[-1][('X','X',0.)])<1e-15


def test_selected_csf_response_and_component_switch(molecule,capsys):
    r=polarizability.main(options(molecule)+['--eps','1e-8','--component','z'])
    assert set(r[-1])=={('Z','Z',0.)}
    assert r[-1][('Z','Z',0.)]==pytest.approx(11.107698996,abs=2e-8)
    capsys.readouterr()
    uncoupled=polarizability.main(options(molecule)+['--eps','1e-8','--component','Z',
                                                   '--no-couple-property','--no-couple-response'])
    assert abs(uncoupled[-1][('Z','Z',0.)])<1e-15


def test_static_polarizability_against_pyscf_finite_field_fci(molecule,capsys):
    from pyscf import gto,scf,fci
    from scsf.drivers.csf_driver import integrals_from_scf
    r=polarizability.main(options(molecule)+['--full','--component','Z'])
    mol=gto.M(atom=molecule,basis='sto-3g',verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-12;mf.kernel()
    ham,nelec,cas=integrals_from_scf(mf,0)
    one=mf.mo_coeff.T@mol.intor('int1e_r',comp=3)[2]@mf.mo_coeff
    eri=ham.two_mo.transpose(0,2,1,3)
    solver=fci.direct_spin0.FCI(mol);solver.conv_tol=1e-13
    def energy(field):
        return solver.kernel(ham.one_mo-field*one,eri,ham.nbasis,nelec,ecore=ham.ecore)[0]
    e0=energy(0.)
    def curvature(step):return -(energy(step)-2*e0+energy(-step))/step**2
    alpha=(4*curvature(.001)-curvature(.002))/3
    npt.assert_allclose(r[-1][('Z','Z',0.)].real,alpha,rtol=0,atol=2e-6)


def test_module_entry_point_and_excited_singlet(molecule,tmp_path):
    run=subprocess.run([sys.executable,'-m','scsf.drivers.polarizability',
        *options(molecule),'--full','--state','1','--component','Z'],
        cwd=Path(__file__).resolve().parents[2],text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'root=1' in run.stdout
    assert float(run.stdout.split('value=')[-1])==pytest.approx(18.674181862,abs=2e-8)
