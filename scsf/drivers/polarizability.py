# SCSF; licensed under GPL-3.0-or-later.
"""Singlet variational SCI polarizabilities in CSF spaces.

Run as ``python -m scsf.drivers.polarizability --xyz examples/water.xyz --basis sto-3g``.
CSFs are the only supported basis.
Triplet property operators and CSF PT2 response are outside this driver.
"""
import argparse
from itertools import count
import numpy as np
import scsf
from ..config import default_cli_threads
from .solvers import solve_ci
from .response import grow_space, resp, one_electron_ao2mo, print_var_summary
from .molecular import MO_SYMMETRY_ATOL, integrals_from_scf, run_scf

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
    parser.add_argument('--point-group', nargs='?', const='auto',
                        help='use point-group blocks, optionally with the named subgroup')
    parser.add_argument('--target-irrep', help='label or code of the CI reference sector')
    parser.add_argument('--nthreads', type=int, default=default_cli_threads(),
                        help='native workers (default: available physical cores, up to 256)')
    parser.add_argument('--davidson-tol', type=float, default=1e-10)
    parser.add_argument('--response-tol', type=float, default=1e-08)
    parser.add_argument('--response-maxiter', type=int, default=200)
    parser.add_argument('--response-growth-tol', type=float, default=0.01)
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.state < 0 or not 1 <= args.nthreads <= 256 or (not np.isfinite(args.eps)) or (args.eps < 0):
        parser.error('require state>=0, threads in 1..256, and finite eps>=0')
    scsf.set_num_threads(args.nthreads)
    group = args.point_group if args.target_irrep is None else args.point_group or 'auto'
    try:
        mf = run_scf(args.xyz, args.basis, args.charge, 0, args.ncore, parser,
                     symmetry=True, point_group=group)
    except ValueError as exc:
        parser.error(str(exc))
    mol = mf.mol
    ham, nelec, cas = integrals_from_scf(mf, args.ncore, packed=True, point_group=group is not None)
    nroots = args.state + 1
    target = args.target_irrep
    if target is not None and target.isdecimal():
        target = int(target)
    try:
        wfn = scsf.csf_wfn(ham.nbasis, nelec, 0, symmetry=ham.symmetry, target_irrep=target)
        dimension = scsf.csf_dimension(ham.nbasis, nelec, 0, symmetry=ham.symmetry, target_irrep=target)
        if nroots > dimension:
            parser.error('requested state exceeds the singlet CSF dimension')
        occupation = (2,)*(nelec//2) + (0,)*(ham.nbasis-nelec//2)
        wfn.add_csf(wfn._reference_in_sector(occupation))
    except ValueError as exc:
        parser.error(str(exc))
    if wfn.target_irrep is not None and len(wfn) < nroots:
        wfn._fill_sector_seed(nroots)
    while len(wfn) < nroots:
        if not scsf.add_csf_one_electron(np.ones((ham.nbasis,) * 2), wfn, np.ones(len(wfn)), eps=0.0, nthreads=args.nthreads):
            parser.error('requested state exceeds the singlet CSF dimension')
    op = scsf.csf_structured_op(ham, wfn, nthreads=args.nthreads)
    def diagonalize(c0):
        energies, vectors, ok = solve_ci(op.matmat, op.diagonal(), roots=nroots, c0=c0, tol=args.davidson_tol, maxiter=500)
        if not ok:
            raise RuntimeError('CI Davidson did not converge')
        return (energies, vectors)
    energies, vectors = diagonalize(None)
    print(f'CSF singlet response: electrons={nelec} orbitals={ham.nbasis} basis={len(wfn)}')
    for iteration in count(1):
        weights = np.max(np.abs(vectors), axis=1)
        added = grow_space(wfn, op, weights, args.eps)
        if not added:
            break
        energies, vectors = diagonalize(np.pad(vectors, ((0, added), (0, 0))))
        print(f'iteration={iteration} added={added} basis={len(wfn)} energy={energies[args.state] + ham.ecore:.12f}')
    for root, energy in enumerate(energies):
        print(f'root={root} energy={energy + ham.ecore:.12f} hartree')
    components = [args.component] if args.component else ['X', 'Y', 'Z']
    dipoles = mol.intor('int1e_r', comp=3)
    operators = {label: one_electron_ao2mo(cas, dipoles['XYZ'.index(label)]) for label in components}
    if ham.symmetry is not None:
        labels = np.asarray(ham.symmetry.orbital_irreps)
        transfers = labels[:, None] ^ labels
        for operator in operators.values():
            for code in np.unique(transfers):
                block = transfers == code
                if np.max(np.abs(operator[block]), initial=0.) <= MO_SYMMETRY_ATOL:
                    operator[block] = 0.
    perturbations = [(label, 0.0, 0) for label in components]
    result = resp(ham, wfn, op, vectors, operators, perturbations, eps_mu=args.eps if args.couple_property else None, eps_resp=args.eps if args.couple_response else None, nroots=nroots, state=args.state, ci_tol=args.davidson_tol, response_tol=args.response_tol, response_maxiter=args.response_maxiter, response_growth_tol=args.response_growth_tol)
    print_var_summary(result[-1])
    return result
def cli():
    main()

if __name__ == '__main__':
    cli()
