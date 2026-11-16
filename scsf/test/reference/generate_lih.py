"""Regenerate the LiH fixture and FCI references using only NumPy and PySCF.

Run from the repository root with ``python scsf/test/reference/generate_lih.py``.
No SCSF code or external integral files are used to produce this fixture.
"""
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, fci, gto, lib, scf


def main():
    lib.num_threads(1)
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', unit='Angstrom', basis='sto-3g',
                spin=0, charge=0, symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF did not converge')
    orbitals = mf.mo_coeff
    one = orbitals.T @ mf.get_hcore() @ orbitals
    one = (one + one.T) / 2
    n = orbitals.shape[1]
    pairs = ao2mo.full(mol, orbitals, aosym='s4', compact=True)
    eri = ao2mo.restore(1, (pairs + pairs.T) / 2, n)
    ecore = mol.energy_nuc()
    references = {}
    for spin in (0, 1):
        solver = fci.direct_spin0.FCI(mol) if spin == 0 else fci.direct_spin1.FCI(mol)
        electrons = (mol.nelectron // 2 + spin, mol.nelectron // 2 - spin)
        energies, vectors = solver.kernel(one, eri, n, electrons, ecore=ecore,
                                          nroots=3, tol=1e-13)
        for vector in vectors:
            ss, _ = fci.spin_op.spin_square(vector, n, electrons)
            np.testing.assert_allclose(ss, spin * (spin + 1), atol=1e-10, rtol=0)
        references['singlet' if spin == 0 else 'triplet'] = energies
    target = Path(__file__).resolve().parents[1] / 'data' / 'lih_sto3g.npz'
    np.savez_compressed(target, ecore=ecore, one=one, two=eri.transpose(0, 2, 1, 3),
                        nelec=mol.nelectron, **references)
    print(f'PySCF {pyscf.__version__}, RHF {mf.e_tot:.14f} Eh')
    for sector, roots in references.items():
        print(f'{sector}: ' + ', '.join(f'{e:.14f}' for e in roots))
    print(target)


if __name__ == '__main__':
    main()
