"""Driver defaults and removal of implicit work-size stopping limits."""
import scsf
import numpy as np
import pytest
from scsf.drivers import csf_driver


def test_compiled_reference_driver_remains_switchable(capsys):
    csf_driver.main(['--integrals','scsf/test/data/lih_sto3g.npz','--coupling-engine','compiled'])
    assert 'energy=-7.882324378883' in capsys.readouterr().out


@pytest.mark.parametrize('full,strategy,product_threads', [
    (False, 'compact-spin-guga', None), (True, 'compact-adaptive-guga', 1),
    (False, 'tuga', 2),
])
def test_direct_driver_full_and_selected_products(full, strategy, product_threads,
                                                 capsys, monkeypatch):
    original = scsf.csf_direct_op
    operators = []
    def direct(*args, **kwargs):
        result = original(*args, **kwargs)
        operators.append(result)
        return result
    def stored(*args, **kwargs):
        pytest.fail('direct calculation constructed a stored Hamiltonian')
    monkeypatch.setattr(scsf, 'csf_direct_op', direct)
    monkeypatch.setattr(scsf, 'csf_op', stored)
    argv = ['--integrals', 'scsf/test/data/lih_sto3g.npz', '--direct',
            '--direct-strategy', strategy, '--nthreads', '2', '--nroots', '2']
    if full:
        argv.append('--full')
    if product_threads is not None:
        argv += ['--product-nthreads', str(product_threads)]
    csf_driver.main(argv)
    output = capsys.readouterr().out
    ground = next(line for line in output.splitlines() if line.startswith('root=0 '))
    assert float(ground.split('energy=')[1].split()[0]) == pytest.approx(-7.882324378883, abs=1e-11)
    assert 'root=1 energy=' in output and f'direct {strategy}' in output
    assert len(operators) == 1
    op = operators[0]
    assert op.nthreads == (2 if product_threads is None else product_threads)
    assert op.shape[0] > 1
    assert op.matrix_cutoff == 0.
    assert op.product_tol == 0.
    assert op.backend.matrix_index == 'zobrist'
    if not full:
        assert 'iteration=1 added=' in output


@pytest.mark.parametrize('options,message', [
    (['--backend', 'reference'], '--direct requires'),
    (['--coupling-engine', 'compiled'], '--direct requires'),
    (['--coupling-engine', 'python'], '--direct requires'),
    (['--product-tol', '-1'], 'product-tol must be'),
    (['--product-tol', 'nan'], 'product-tol must be'),
])
def test_direct_driver_invalid_options_fail_before_integrals(options, message, capsys):
    with pytest.raises(SystemExit) as error:
        csf_driver.main(['--integrals', 'nonexistent.npz', '--direct', *options])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


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
    # Independent RHF energy checks the restored tensor's index convention,
    # frozen-core contribution, and acceptance by both coupling backends.
    for backend in (scsf.CSFReferenceBackend(ham),
                    scsf.CSFSpinBackend(ham,coupling_engine='guga',cache_bytes=0)):
        energy=backend.block([label],[label],0)[0,0]+ham.ecore
        assert energy==pytest.approx(seen['mf'].e_tot,abs=2e-11,rel=0)
