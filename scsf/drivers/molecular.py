# SCSF; licensed under GPL-3.0-or-later.
"""Build active-space integrals for the energy and response drivers."""
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


def integrals_from_scf(mf, ncore):
    """Common real spatial integrals and CAS metadata for CI/property drivers."""
    from pyscf import ao2mo, mcscf
    mol = mf.mol
    if not 0 <= ncore <= min(mol.nelec):
        raise ValueError('ncore must leave a valid active electron count')
    ncas = mf.mo_coeff.shape[1] - ncore
    nelec = mol.nelectron - 2 * ncore
    cas = mcscf.CASCI(mf, ncas, nelec)
    h1, ecore = cas.get_h1eff()
    h1 = 0.5 * (h1 + h1.T)
    mo = mf.mo_coeff[:, ncore:ncore + ncas]
    pairs = ao2mo.full(mol, mo, aosym='s4', compact=True)
    eri = ao2mo.restore(1, 0.5 * (pairs + pairs.T), ncas)
    return (scsf.hamiltonian(ecore, h1, eri.transpose(0, 2, 1, 3)), nelec, cas)
