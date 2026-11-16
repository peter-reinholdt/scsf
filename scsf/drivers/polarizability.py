# SCSF; licensed under GPL-3.0-or-later.
"""Singlet variational SCI polarizabilities in CSF spaces.

Run as ``python -m benchmarks.polarizability --xyz examples/water.xyz --basis sto-3g``.
CSFs are the only supported basis.
Triplet property operators and CSF PT2 response are outside this driver.
"""
import argparse
from itertools import count
import numpy as np
import scsf
from .solvers import solve_ci
from .response import resp, wrap_matvec, one_electron_ao2mo, print_var_summary
from .csf_driver import integrals_from_scf

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xyz', required=True)
    parser.add_argument('--basis', required=True)
    parser.add_argument('--ncore', type=int, default=0)
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--couple-property', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--couple-response', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--component', type=str.upper, choices=('X', 'Y', 'Z'))
    parser.add_argument('--state', type=int, default=0)
    parser.add_argument('--full', action='store_true', help='full selected-spin space for small checks')
    parser.add_argument('--nthreads', type=int, default=4)
    parser.add_argument('--davidson-tol', type=float, default=1e-10)
    parser.add_argument('--response-tol', type=float, default=1e-08)
    parser.add_argument('--response-maxiter', type=int, default=200)
    parser.add_argument('--response-growth-tol', type=float, default=0.01)
    return parser

def main(argv=None):
    from pyscf import gto, scf
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.state < 0 or not 1 <= args.nthreads <= 256 or (not np.isfinite(args.eps)) or (args.eps < 0):
        parser.error('require state>=0, threads in 1..256, and finite eps>=0')
    scsf.set_num_threads(args.nthreads)
    mol = gto.M(atom=args.xyz, basis=args.basis, charge=args.charge, spin=0, symmetry=True, verbose=0)
    if not 0 <= args.ncore <= min(mol.nelec):
        parser.error('--ncore must leave a valid active electron count')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('orbital calculation failed to converge')
    ham, nelec, cas = integrals_from_scf(mf, args.ncore)
    nroots = args.state + 1
    wfn = scsf.csf_wfn(ham.nbasis, nelec, 0)
    backend = scsf.CSFSpinBackend(ham, coupling_engine='guga', guga_factors='spectator', nthreads=args.nthreads, cache_bytes=0, selection_engine='ordered', matrix_index='residue', coupling_reuse=True, reuse_strategy='packed', walk_engine='counted', parallel_engine='shared', matrix_parallel_engine='shared', max_tensor_entries=None, max_bond_elements=None, max_connection_terms=None)
    if args.full:
        wfn.add_all_csfs(max_csfs=None)
    else:
        wfn.add_csf('2' * (nelec // 2) + '0' * (ham.nbasis - nelec // 2))
        while len(wfn) < nroots:
            if not scsf.add_csf_one_electron(np.ones((ham.nbasis,) * 2), wfn, np.ones(len(wfn)), eps=0.0, nthreads=args.nthreads):
                parser.error('requested state exceeds the singlet CSF dimension')
    op = scsf.csf_op(ham, wfn, backend=backend, max_matrix_entries=None, product_mode='native')
    if len(wfn) < nroots:
        parser.error('requested state exceeds the initial space')
    def diagonalize(c0):
        energies, vectors, ok = solve_ci(lambda x: wrap_matvec(op, x), op.diagonal(), roots=nroots, c0=c0, tol=args.davidson_tol, maxiter=500)
        if not ok:
            raise RuntimeError('CI Davidson did not converge')
        return (energies, vectors)
    energies, vectors = diagonalize(None)
    print(f'CSF singlet response: electrons={nelec} orbitals={ham.nbasis} basis={len(wfn)}')
    if not args.full:
        for iteration in count(1):
            weights = np.max(np.abs(vectors), axis=1)
            added = scsf.add_csf_hci(ham, wfn, weights, eps=args.eps, backend=backend, max_csfs=None, max_candidates=None, max_configurations=None)
            if not added:
                break
            op.update(ham, wfn)
            energies, vectors = diagonalize(np.pad(vectors, ((0, added), (0, 0))))
            print(f'iteration={iteration} added={added} basis={len(wfn)} energy={energies[args.state] + ham.ecore:.12f}')
    for root, energy in enumerate(energies):
        print(f'root={root} energy={energy + ham.ecore:.12f} hartree')
    components = [args.component] if args.component else ['X', 'Y', 'Z']
    dipoles = mol.intor('int1e_r', comp=3)
    operators = {label: one_electron_ao2mo(cas, dipoles['XYZ'.index(label)]) for label in components}
    perturbations = [(label, 0.0, 0) for label in components]
    result = resp(ham, wfn, op, vectors, operators, perturbations, eps_mu=args.eps if args.couple_property else None, eps_resp=args.eps if args.couple_response else None, nroots=nroots, state=args.state, ci_tol=args.davidson_tol, response_tol=args.response_tol, response_maxiter=args.response_maxiter, response_growth_tol=args.response_growth_tol)
    print_var_summary(result[-1])
    return result
def cli():
    main()

if __name__ == '__main__':
    cli()
