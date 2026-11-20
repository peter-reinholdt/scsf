"""CLI integration for the native kernels and molecular integral input."""
import numpy as np
import pytest
import scsf
from scsf import config
from scsf.drivers import csf_driver, polarizability


LIH = 'scsf/test/data/lih_sto3g.npz'


@pytest.mark.parametrize('ncore',[0,1])
def test_diffuse_xyz_integrals_have_exact_real_symmetry(ncore,monkeypatch):
    pytest.importorskip('pyscf')
    from pyscf import scf
    original=scf.RHF
    seen={}
    def record_mf(mol):
        seen['mf']=original(mol)
        return seen['mf']
    monkeypatch.setattr(scf,'RHF',record_mf)
    parser=csf_driver.build_parser()
    args=parser.parse_args(['--xyz','examples/water.xyz','--basis','aug-cc-pvdz',
                            '--ncore',str(ncore)])
    ham,nelec=csf_driver.integrals(args,parser)
    h,g=ham.one_mo,ham.two_mo.transpose(0,2,1,3)
    np.testing.assert_array_equal(h,h.T)
    for axes in ((1,0,2,3),(0,1,3,2),(2,3,0,1)):
        np.testing.assert_array_equal(g,g.transpose(axes))
    assert ham.nbasis==41-ncore and nelec==10-2*ncore
    label='2'*(nelec//2)+'0'*(ham.nbasis-nelec//2)
    wfn = scsf.csf_wfn(ham.nbasis, nelec, 0)
    wfn.add_csf(label)
    op = scsf.csf_structured_op(ham, wfn, strategy='cih')
    assert op.diagonal()[0] + ham.ecore == pytest.approx(seen['mf'].e_tot, abs=2e-11, rel=0)


@pytest.mark.parametrize('kernel', scsf.csf_structured_op.kernels)
def test_driver_selected(kernel, capsys):
    csf_driver.main(['--integrals', LIH, '--kernel', kernel, '--nthreads', '2', '--nroots', '2'])
    output = capsys.readouterr().out
    ground = next(line for line in output.splitlines() if line.startswith('root=0 '))
    assert float(ground.split('energy=')[1].split()[0]) == pytest.approx(-7.882324378883, abs=1e-11)
    assert f'{kernel}, csf HCI' in output and 'root=1 energy=' in output
    assert 'iteration=1 added=' in output


def test_default_nminus2_without_main_work_array(capsys):
    csf_driver.main(['--integrals', LIH, '--nthreads', '1', '--max-product-scratch-bytes', '0'])
    assert 'nminus2, csf HCI' in capsys.readouterr().out


def test_configuration_hci_driver(capsys):
    csf_driver.main(['--integrals', LIH, '--kernel', 'cih', '--selection-unit', 'configuration', '--sci-stats'])
    output = capsys.readouterr().out
    assert 'cih, configuration HCI' in output and 'ncsf=41' in output
    assert "selection {'added': 0" in output


@pytest.mark.parametrize('flag', [
    '--max-expansion-nnz', '--direct', '--backend', '--product-engine', '--full',
    '--frozen-basis', '--expected-products', '--cih-slot-sweep', '--davidson-olsen',
    '--davidson-trial-vector-tol', '--davidson-trial-vector-decay', '--davidson-trial-vector-source',
])
def test_legacy_flags_are_gone(flag):
    assert flag not in csf_driver.build_parser().format_help()


def test_automatic_kernel_is_rejected(capsys):
    with pytest.raises(SystemExit, match='2'):
        csf_driver.build_parser().parse_args(['--integrals', LIH, '--kernel', 'auto'])
    assert "invalid choice: 'auto'" in capsys.readouterr().err


def test_cli_defaults_count_physical_cores_within_affinity(tmp_path, monkeypatch):
    for cpu, siblings in ((1, '1,3'), (3, '1,3'), (7, '7,9')):
        topology = tmp_path / f'cpu{cpu}' / 'topology'
        topology.mkdir(parents=True)
        (topology / 'thread_siblings_list').write_text(siblings + '\n')
    monkeypatch.setattr(config, '_CPU_SYSFS', tmp_path)
    monkeypatch.setattr(config.os, 'sched_getaffinity', lambda _: {1, 3, 7}, raising=False)
    assert csf_driver.build_parser().parse_args(['--integrals', LIH]).nthreads == 2
    args = polarizability.build_parser().parse_args(['--xyz', 'water.xyz', '--basis', 'sto-3g'])
    assert args.nthreads == 2


@pytest.mark.parametrize('unit, expected', [('csf', {'uudd00'}), ('configuration', {'uudd00', 'udud00'})])
def test_reference_is_padded_before_initial_space_selection(unit, expected):
    parser = csf_driver.build_parser()
    args = parser.parse_args(['--integrals', LIH, '--reference', 'uudd', '--selection-unit', unit])
    ham, nelec = csf_driver.integrals(args, parser)
    wfn, _ = csf_driver.initial_space(args, parser, ham, nelec)
    assert set(wfn) == expected


def test_reference_longer_than_active_space_is_rejected(capsys):
    with pytest.raises(SystemExit, match='2'):
        csf_driver.main(['--integrals', LIH, '--reference', '2200000'])
    assert 'invalid --reference' in capsys.readouterr().err


def test_compact_snapshot_and_exact_leading_block(tmp_path, capsys):
    with np.load(LIH) as data:
        ham = scsf.Hamiltonian(float(data['ecore']), data['one'], data['two'])
        compact = scsf.PackedHamiltonian.from_dense(ham)
        snapshot = tmp_path / 'packed.npz'
        np.savez(snapshot, ecore=ham.ecore, one=ham.one_mo, packed_eri=compact.packed_eri,
                 nelec=int(data['nelec']))
    csf_driver.main(['--integrals', str(snapshot), '--kernel', 'nminus2', '--davidson-block-size', '4'])
    ground = next(line for line in capsys.readouterr().out.splitlines() if line.startswith('root=0 '))
    assert float(ground.split('energy=')[1].split()[0]) == pytest.approx(-7.882324378883, abs=1e-11)
