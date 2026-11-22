# SCSF; licensed under GPL-3.0-or-later.
"""Build active-space integrals for the energy and response drivers."""
import numpy as np

import scsf


def run_scf(xyz, basis, charge, two_s, ncore, parser, *, symmetry=False):
    from pyscf import gto, scf

    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=two_s, symmetry=symmetry, verbose=0)
    if not 0 <= ncore <= min(mol.nelec):
        parser.error('--ncore must leave a valid active electron count')
    mf = scf.RHF(mol) if two_s == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('orbital calculation failed to converge')
    return mf


def correlated_density(mf, ncore, method):
    """Spin-summed MP2 or CCSD 1-RDM in the original active spatial basis."""
    from pyscf import cc, mp

    active = mf.mo_coeff[:, ncore:]
    if all(n in (ncore, mf.mo_coeff.shape[1]) for n in mf.mol.nelec):
        return np.diag(mf.mo_occ[ncore:])

    reference = mf.to_uhf() if mf.mol.spin else mf
    solver = (mp.MP2 if method == 'mp2' else cc.CCSD)(reference, frozen=ncore or None)
    eris = solver.ao2mo()
    solver.kernel(eris=eris)
    if method == 'ccsd':
        if not solver.converged:
            raise RuntimeError('CCSD failed to converge for natural orbitals')
        solver.solve_lambda(eris=eris)
        if not solver.converged_lambda:
            raise RuntimeError('CCSD Lambda equations failed to converge for natural orbitals')
    density = np.asarray(solver.make_rdm1(ao_repr=True))
    if density.ndim == 3:
        density = density.sum(axis=0)
    projection = mf.get_ovlp() @ active
    return projection.T @ density @ projection


def integrals_from_scf(mf, ncore, *, mo_coeff=None):
    """Common real spatial integrals and CAS metadata for CI/property drivers."""
    from pyscf import ao2mo, mcscf
    mol = mf.mol
    if not 0 <= ncore <= min(mol.nelec):
        raise ValueError('ncore must leave a valid active electron count')
    ncas = mf.mo_coeff.shape[1] - ncore
    nelec = mol.nelectron - 2 * ncore
    cas = mcscf.CASCI(mf, ncas, nelec)
    if mo_coeff is not None:
        cas.mo_coeff = mo_coeff
    h1, ecore = cas.get_h1eff()
    h1 = 0.5 * (h1 + h1.T)
    mo = cas.mo_coeff[:, ncore:ncore + ncas]
    pairs = ao2mo.full(mol, mo, aosym='s4', compact=True)
    eri = ao2mo.restore(1, 0.5 * (pairs + pairs.T), ncas)
    return (scsf.hamiltonian(ecore, h1, eri.transpose(0, 2, 1, 3)), nelec, cas)
