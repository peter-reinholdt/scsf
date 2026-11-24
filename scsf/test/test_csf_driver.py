"""CLI integration for the native kernels and molecular integral input."""
import numpy as np
import pytest
import scsf
from scsf import config
from scsf.csf import _configuration
from scsf.drivers import csf_driver, polarizability
from scsf.integrals import rotate_hamiltonian


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


def test_default_nminus2_with_global_memory_limit(capsys):
    csf_driver.main(['--integrals', LIH, '--nthreads', '1', '--max-mem', '1MiB'])
    assert 'nminus2, csf HCI' in capsys.readouterr().out


def test_insufficient_global_cap_reports_setup_error(capsys):
    with pytest.raises(SystemExit, match='1'):
        csf_driver.main(['--integrals', LIH, '--kernel', 'cih', '--max-mem', '1', '--nthreads', '1'])
    assert 'cih setup with max-mem=1 bytes' in capsys.readouterr().err


def test_configuration_hci_driver(capsys):
    csf_driver.main(['--integrals', LIH, '--kernel', 'cih', '--selection-unit', 'configuration', '--sci-stats'])
    output = capsys.readouterr().out
    assert 'cih, configuration HCI' in output and 'ncsf=41' in output
    assert "selection {'added': 0" in output


def test_driver_solves_cutoff_batch_and_stops_selection(tmp_path, capsys):
    # The reference connects only to row 1. Solving those two rows gives
    # enough weight on row 1 to select row 2 in a subsequent HCI step.
    one = np.diag([0., 1., 2.])
    one[0, 1] = one[1, 0] = one[1, 2] = one[2, 1] = .125
    path = tmp_path/'chain.npz'
    np.savez(path, ecore=0., nelec=1, one=one, two=np.zeros((3,)*4))
    args = ['--integrals', str(path), '--two-s', '1', '--eps', '.01',
            '--nthreads', '1', '--selection-unit', 'configuration']
    csf_driver.main(args + ['--stop-growth-fraction', '1'])
    output = capsys.readouterr().out
    assert 'Final Davidson solve for 2 CSFs' in output
    assert 'Davidson iter' in output and 'wall=' in output
    assert 'max Ritz residual=' in output
    assert 'Hamiltonian products inside Davidson' in output
    assert output.count('Selecting from') == 1
    assert 'iteration=1 added=1 ncsf=2' in output
    energy = np.linalg.eigvalsh(one[:2, :2])[0]
    assert f'root=0 energy={energy:.12f}' in output

    csf_driver.main(args + ['--stop-growth-fraction', '0', '--quiet'])
    strict = capsys.readouterr().out
    assert 'iteration=2 added=1 ncsf=3' in strict
    assert 'Davidson iter' not in strict


def test_diagonalize_reports_ritz_residuals_without_extra_products(monkeypatch):
    parser = csf_driver.build_parser()
    args = parser.parse_args(['--integrals', LIH, '--nthreads', '1', '--nroots', '2'])
    ham, nelec = csf_driver.integrals(args, parser)
    wfn, _ = csf_driver.initial_space(args, parser, ham, nelec)
    wfn.add_all_csfs()
    op = csf_driver.make_operator(args, ham, wfn)
    native_matmat = op.native_matmat
    calls = []

    def record_product(vectors):
        calls.append(vectors.shape[1])
        return native_matmat(vectors)

    monkeypatch.setattr(op, 'native_matmat', record_product)
    measurements = {}
    energies, vectors = csf_driver.diagonalize(args, op, measurements=measurements)
    assert calls == measurements['solver']['call_widths']
    residuals = np.linalg.norm(native_matmat(vectors) - vectors * (energies - op.ecore), axis=0)
    np.testing.assert_allclose(measurements['residuals'], residuals, atol=1e-13, rtol=0)


@pytest.mark.parametrize('unit', ['csf', 'configuration'])
def test_growth_cutoff_leaves_solved_space_unchanged(unit):
    parser = csf_driver.build_parser()
    args = parser.parse_args(['--integrals', LIH, '--selection-unit', unit, '--nthreads', '1'])
    ham, nelec = csf_driver.integrals(args, parser)
    wfn, _ = csf_driver.initial_space(args, parser, ham, nelec)
    wfn.add_all_csfs()
    # Omit one occupation to leave a small nonzero append.
    labels = list(wfn)
    omitted = _configuration(labels[-1])
    space = scsf.csf_wfn(ham.nbasis, nelec, args.two_s)
    for label in labels:
        if _configuration(label) != omitted:
            space.add_csf(label)
    op = csf_driver.make_operator(args, ham, space)
    old_labels = list(op.labels(native=True))
    _, c = csf_driver.diagonalize(args, op)
    omitted_count = len(labels)-len(old_labels)
    cutoff = omitted_count/len(old_labels)
    remap, stopped = op.select_hci(abs(c[:, 0]), 0., selection_unit=unit, stop_growth_fraction=cutoff)
    assert remap.size == 0 and stopped['added'] == 0 and stopped['proposed_csfs'] > 0
    assert 'basis_extend_ns' not in stopped
    assert list(op.labels(native=True)) == old_labels
    _, strict = op.select_hci(abs(c[:, 0]), 0., selection_unit=unit,
                              stop_growth_fraction=np.nextafter(cutoff, 0.))
    assert strict['added'] == stopped['proposed_csfs']


@pytest.mark.parametrize('flag', [
    '--max-expansion-nnz', '--direct', '--backend', '--product-engine', '--full',
    '--frozen-basis', '--expected-products', '--cih-slot-sweep', '--davidson-olsen',
    '--davidson-trial-vector-tol', '--davidson-trial-vector-decay', '--davidson-trial-vector-source',
    '--max-ordered-bytes', '--max-product-scratch-bytes', '--max-plan-bytes',
    '--max-structured-metadata-bytes', '--max-structured-coefficient-bytes',
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


def test_sci_natural_orbitals_rotate_the_final_reference(tmp_path, capsys):
    one = np.array([[0., .25], [.25, 1.]])
    snapshot = tmp_path / 'one-electron.npz'
    np.savez(snapshot, ecore=0., nelec=1, one=one, two=np.zeros((2,) * 4))
    csf_driver.main(['--integrals', str(snapshot), '--two-s', '1', '--natorb',
                     '--nthreads', '1', '--quiet'])
    pilot, final = capsys.readouterr().out.split('HCI run in natural orbitals')
    energy = np.linalg.eigvalsh(one)[0]
    assert 'iteration=1 added=1 ncsf=2' in pilot
    occupations = pilot.split('SCI natural occupations=[')[1].split(']')[0]
    np.testing.assert_allclose(np.fromstring(occupations, sep=' '), [1., 0.], atol=1e-8)
    assert f'iteration=0 ncsf=1 energies=[{energy:.12f}]' in final
    assert f'root=0 energy={energy:.12f}' in final


def test_natural_orbital_rotation_preserves_packed_eris():
    rng = np.random.default_rng(59)
    x = rng.normal(size=(4, 6))
    a = rng.normal(size=(6, 6))
    core = a @ a.T
    two = np.einsum('pu,qu,uv,rv,sv->prqs', x, x, core, x, x, optimize=True)
    one = np.diag([-.6, 1., 2., 3.])
    dense = scsf.Hamiltonian(.5, one, two)
    ham = scsf.PackedHamiltonian.from_dense(dense)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    transformed = rotate_hamiltonian(ham, rotation)
    assert type(transformed) is type(ham)
    assert transformed.ecore == ham.ecore
    wfn = scsf.csf_wfn(4, 4, 0)
    wfn.add_all_csfs()
    basis = np.eye(len(wfn))
    actual = scsf.csf_structured_op(transformed, wfn, nthreads=1).matmat(basis)
    expected = scsf.csf_structured_op(rotate_hamiltonian(dense, rotation), wfn, nthreads=1).matmat(basis)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    original = scsf.csf_structured_op(dense, wfn, nthreads=1).matmat(basis)
    np.testing.assert_allclose(np.linalg.eigvalsh(actual), np.linalg.eigvalsh(original), rtol=0, atol=1e-9)


@pytest.mark.parametrize('method', ['mp2', 'ccsd'])
def test_molecular_natural_orbitals_honor_frozen_core(method, capsys):
    pytest.importorskip('pyscf')
    csf_driver.main(['--xyz', 'examples/water.xyz', '--basis', 'sto-3g', '--ncore', '1',
                     '--natorb', method, '--nthreads', '1', '--eps', '0', '--davidson-tol', '1e-10', '--quiet'])
    output = capsys.readouterr().out
    occupations = next(line.split('[')[1].rstrip(']') for line in output.splitlines()
                       if line.startswith(f'{method.upper()} natural occupations='))
    values = list(map(float, occupations.split()))
    assert len(values) == 6
    assert values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(8., abs=1e-7, rel=0)
    assert 0. < values[-1] < .1
    ground = next(line for line in output.splitlines() if line.startswith('root=0 '))
    assert float(ground.split('energy=')[1].split()[0]) == pytest.approx(-75.012870400157, abs=1e-10, rel=0)


@pytest.mark.parametrize('method', ['mp2', 'ccsd'])
def test_correlated_natural_orbitals_require_xyz(method, capsys):
    with pytest.raises(SystemExit, match='2'):
        csf_driver.main(['--integrals', LIH, '--natorb', method])
    assert f'--natorb {method} requires --xyz' in capsys.readouterr().err
