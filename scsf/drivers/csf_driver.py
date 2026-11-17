# SCSF; licensed under GPL-3.0-or-later.
"""Native or reference CSF CI/HCI with Davidson diagonalization.

Examples (run from the repository with SCSF importable)::

    python -m scsf --integrals scsf/test/data/lih_sto3g.npz --full
    python -m scsf --integrals scsf/test/data/lih_sto3g.npz --eps 1e-4
    python -m scsf --xyz examples/water.xyz --basis sto-3g --eps 1e-4
    python -m scsf --xyz examples/water.xyz --basis cc-pvdz --ncore 1 --direct

The native spin backend is the default; --backend reference selects the
determinant oracle. PySCF is required only for the XYZ input route.
Stored CSR products remain the default; --direct uses compact-spin-guga products.
"""
import argparse
from itertools import count
import numpy as np
import scsf
_DIRECT_STRATEGIES = scsf.csf_direct_op._strategies
from .solvers import solve_ci

def solve_variational(op, nroots, c0=None, *, tol=1e-10, maxiter=200, max_subspace=None, trial_vector_tol=0.0, trial_vector_decay=0.0, trial_vector_source='orthogonal'):
    """Davidson CI with column vectors; add the core energy exactly once."""
    energies, coeffs, converged = solve_ci(op.matmat, op.diagonal(), roots=nroots, c0=c0, tol=tol, maxiter=maxiter, max_subspace=max_subspace, trial_vector_tol=trial_vector_tol, trial_vector_decay=trial_vector_decay, trial_vector_source=trial_vector_source)
    if not converged:
        raise RuntimeError('Davidson did not converge; increase --davidson-maxiter or --davidson-max-subspace')
    return (energies + op.ecore, coeffs)

def integrals(args, parser):
    if args.integrals:
        if args.ncore:
            parser.error('--ncore applies to XYZ input; snapshots must already contain active integrals')
        with np.load(args.integrals, allow_pickle=False) as data:
            nelec = args.nelec if args.nelec is not None else int(data['nelec']) if 'nelec' in data else None
            if nelec is None:
                parser.error('--nelec is required if the array snapshot has no nelec field')
            return (scsf.hamiltonian(float(data['ecore']), data['one'], data['two']), nelec)
    if not args.basis:
        parser.error('--basis is required with --xyz')
    if args.nelec is not None:
        parser.error('--nelec applies to array snapshot input; XYZ uses the molecular charge')
    from pyscf import gto, scf
    mol = gto.M(atom=args.xyz, basis=args.basis, charge=args.charge, spin=args.two_s, symmetry=False, verbose=0)
    if not 0 <= args.ncore <= min(mol.nelec):
        parser.error('--ncore must leave a valid active electron count')
    mf = scf.RHF(mol) if args.two_s == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('orbital calculation failed to converge')
    ham, nelec, _ = integrals_from_scf(mf, args.ncore)
    return (ham, nelec)

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

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--integrals')
    source.add_argument('--xyz')
    parser.add_argument('--basis')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--ncore', type=int, default=0)
    parser.add_argument('--nelec', type=int)
    parser.add_argument('--two-s', type=int, default=0, help='twice the target total spin')
    parser.add_argument('--reference', help='explicit path, e.g. uudd or 22ud00')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--nroots', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.0001)
    parser.add_argument('--max-iterations', type=int, help='optional SCI iteration cap; default: unlimited')
    parser.add_argument('--max-csfs', type=int, help='optional CSF cap; default: unlimited')
    parser.add_argument('--max-candidates', type=int, help='optional candidate cap; default: unlimited')
    parser.add_argument('--max-configurations', type=int, help='optional configuration cap; default: unlimited')
    parser.add_argument('--max-matrix-entries', type=int, help='optional stored-nnz cap; default: unlimited')
    parser.add_argument('--max-expansion-nnz', type=int, help='optional reference-expansion cap; default: unlimited')
    parser.add_argument('--max-block-elements', type=int, help='optional dense-block cap; default: unlimited')
    parser.add_argument('--backend', choices=('spin', 'reference'), default='spin')
    parser.add_argument('--coupling-engine', choices=('compiled', 'python', 'guga'), default='guga')
    parser.add_argument('--max-transfer-bounds', type=int, default=1000000)
    parser.add_argument('--selection-engine', choices=('reference', 'ordered'), default='ordered')
    parser.add_argument('--parallel-engine', choices=('reference', 'shared'), default='shared')
    parser.add_argument('--matrix-parallel-engine', choices=('reference', 'shared', 'private'), default='shared', help='matrix scheduling, independent of selection (default: shared)')
    parser.add_argument('--matrix-index', choices=('reference', 'trie', 'residue', 'zobrist'), help='occupation index (default: zobrist for direct, residue for stored CSR)')
    parser.add_argument('--guga-factors', choices=('reference', 'table', 'spectator'), help='default: spectator for GUGA, reference otherwise')
    parser.add_argument('--coupling-reuse', action=argparse.BooleanOptionalAction, default=None, help='default: enabled for GUGA')
    parser.add_argument('--collect-reuse', action='store_true')
    parser.add_argument('--max-ordered-bytes', type=int, default=16 * 1024 ** 2)
    parser.add_argument('--max-index-bytes', type=int, default=64 * 1024 ** 2)
    parser.add_argument('--max-reuse-bytes', type=int, default=16 * 1024 ** 2)
    parser.add_argument('--max-reuse-bytes-per-worker', type=int, help='explicit worker cache cap; overrides the total --max-reuse-bytes')
    parser.add_argument('--matrix-kernel', choices=('general', 'ranked'), default='general', help='CSR coupling evaluator; ranked reuses the direct tiny spin cache')
    parser.add_argument('--reuse-strategy', choices=('string', 'packed'), default='packed')
    parser.add_argument('--walk-engine', choices=('reference', 'connected', 'adaptive', 'counted'), help='default: counted for GUGA, reference otherwise')
    parser.add_argument('--max-walk-bytes', type=int, default=4 * 1024 ** 2)
    parser.add_argument('--selection', choices=('auto', 'native', 'python'), default='auto')
    parser.add_argument('--selection-rule', choices=('hci', 'configuration', 'bound', 'transfer'), default='hci')
    parser.add_argument('--parent-fraction', type=float, default=1.0, help='experimental parent occupation norm fraction in (0,1]; default: all parents')
    parser.add_argument('--no-screening', action='store_true')
    parser.add_argument('--no-prescreen', action='store_true', help='disable the cheap transfer bound; retain prepared bounds')
    parser.add_argument('--sci-stats', action='store_true')
    parser.add_argument('--nthreads', type=int, default=4, help='native configuration workers (default: 4)')
    parser.add_argument('--direct', action='store_true', help='apply Hamiltonian products directly (requires spin backend and GUGA)')
    parser.add_argument('--direct-strategy', choices=_DIRECT_STRATEGIES, default='compact-spin-guga', help='strategy with --direct (default: compact-spin-guga)')
    parser.add_argument('--product-tol', type=float, default=0.0, help='direct product screening tolerance (default: 0, unscreened)')
    parser.add_argument('--max-product-scratch-bytes', type=int, default=256*1024**2,
                        help='direct output/omission scratch ceiling; may reduce workers or tile RHS columns')
    parser.add_argument('--product-accumulation', choices=('private','gather'), default='private',
                        help='experimental exact compact gather owns disjoint output rows')
    parser.add_argument('--spin-encoder', choices=('auto','portable','bmi2'), default='auto')
    parser.add_argument('--compact-spin-kernel', choices=('general','short'), default='general')
    parser.add_argument('--worker-cpus', type=int, nargs='+', help='one allowed Linux CPU per adaptive product worker')
    parser.add_argument('--product-engine', choices=('native', 'scipy'), default='native', help='stored CSR product engine (default: native)')
    parser.add_argument('--matrix-cutoff', type=float, help='off-diagonal cutoff (default: 0 for direct, 1e-10 for stored CSR)')
    parser.add_argument('--storage-engine', choices=('auto', 'native', 'snapshot'), default='auto', help='native append storage for symmetric native products; snapshot keeps the reference')
    parser.add_argument('--product-nthreads', type=int, help='product threads (default: native worker count)')
    parser.add_argument('--product-mode', choices=('native', 'native-gather'), default='native')
    parser.add_argument('--cache-mib', type=int, default=0, help='retained native coupling cache in MiB; 0 disables retention')
    parser.add_argument('--davidson-tol', type=float, default=1e-10)
    parser.add_argument('--davidson-maxiter', type=int, default=200)
    parser.add_argument('--davidson-max-subspace', type=int)
    parser.add_argument('--davidson-trial-vector-tol', type=float, default=0.0, help='experimental normalized correction cutoff; zero disables clamping')
    parser.add_argument('--davidson-trial-vector-decay', type=float, default=0.0, help='cutoff taper exponent versus relative residual; zero keeps a fixed cutoff')
    parser.add_argument('--davidson-trial-vector-source', choices=('orthogonal', 'preconditioned'), default='orthogonal', help='correction to clamp (default: orthogonal)')
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.direct and (args.backend != 'spin' or args.coupling_engine != 'guga'):
        parser.error('--direct requires --backend spin and --coupling-engine guga')
    if not np.isfinite(args.product_tol) or args.product_tol < 0:
        parser.error('product-tol must be finite and nonnegative')
    if args.max_product_scratch_bytes < 0:
        parser.error('max-product-scratch-bytes must be nonnegative')
    if args.matrix_cutoff is None:
        args.matrix_cutoff = 0.0 if args.direct else 1e-10
    if args.matrix_index is None:
        args.matrix_index = 'zobrist' if args.direct else 'residue'
    guga = args.coupling_engine == 'guga'
    if args.guga_factors is None:
        args.guga_factors = 'spectator' if guga else 'reference'
    if args.coupling_reuse is None:
        args.coupling_reuse = guga
    if args.walk_engine is None:
        args.walk_engine = 'counted' if guga else 'reference'
    if args.nroots < 1 or (args.max_iterations is not None and args.max_iterations < 1) or (not np.isfinite(args.eps)) or (args.eps < 0):
        parser.error('require nroots/max-iterations >= 1 and finite eps >= 0')
    if not np.isfinite(args.parent_fraction) or not 0 < args.parent_fraction <= 1:
        parser.error('parent-fraction must be finite and in (0,1]')
    ham, nelec = integrals(args, parser)
    wfn = scsf.csf_wfn(ham.nbasis, nelec, args.two_s)
    dimension = scsf.csf_dimension(ham.nbasis, nelec, args.two_s)
    if args.nroots > dimension:
        parser.error('nroots exceeds the target-spin dimension')
    if args.full:
        wfn.add_all_csfs(max_csfs=args.max_csfs)
    else:
        pairs = (nelec - args.two_s) // 2
        reference = args.reference if args.reference is not None else '2' * pairs + 'u' * args.two_s + '0' * (ham.nbasis - pairs - args.two_s)
        wfn.add_csf(reference)
    if args.backend == 'reference':
        backend = scsf.CSFReferenceBackend(ham, max_expansion_nnz=args.max_expansion_nnz, max_block_elements=args.max_block_elements)
    else:
        backend = scsf.CSFSpinBackend(ham, max_block_elements=args.max_block_elements, max_bond_elements=None, max_tensor_entries=None, max_connection_terms=None, nthreads=args.nthreads, cache_bytes=args.cache_mib * 1024 ** 2, prescreen=not args.no_prescreen, coupling_engine=args.coupling_engine, max_transfer_bounds=args.max_transfer_bounds, selection_engine=args.selection_engine, matrix_index=args.matrix_index, guga_factors=args.guga_factors, coupling_reuse=args.coupling_reuse, collect_reuse=args.collect_reuse, max_ordered_bytes=args.max_ordered_bytes, max_index_bytes=args.max_index_bytes, max_reuse_bytes=args.max_reuse_bytes, reuse_strategy=args.reuse_strategy, walk_engine=args.walk_engine, max_walk_bytes=args.max_walk_bytes, parallel_engine=args.parallel_engine, matrix_parallel_engine=args.matrix_parallel_engine, matrix_kernel=args.matrix_kernel, max_reuse_bytes_per_worker=args.max_reuse_bytes_per_worker)
    if args.direct:
        op = scsf.csf_direct_op(ham, wfn, backend=backend, strategy=args.direct_strategy,
            product_tol=args.product_tol, matrix_cutoff=args.matrix_cutoff, nthreads=args.product_nthreads,
            max_ordered_bytes=args.max_ordered_bytes,
            max_product_scratch_bytes=args.max_product_scratch_bytes,
            product_accumulation=args.product_accumulation, spin_encoder=args.spin_encoder,
            compact_spin_kernel=args.compact_spin_kernel, worker_cpus=args.worker_cpus,
            collect_spin_stats=args.sci_stats or args.direct_strategy == 'semi-direct')
    else:
        op = scsf.csf_op(ham, wfn, backend=backend, product_engine=args.product_engine, matrix_cutoff=args.matrix_cutoff, storage_engine=args.storage_engine, product_nthreads=args.product_nthreads, product_mode=args.product_mode, max_matrix_entries=args.max_matrix_entries)
    solver_options = dict(tol=args.davidson_tol, maxiter=args.davidson_maxiter, trial_vector_tol=args.davidson_trial_vector_tol, trial_vector_decay=args.davidson_trial_vector_decay, trial_vector_source=args.davidson_trial_vector_source)

    def diagonalize(c0=None):
        limit = args.davidson_max_subspace
        if limit is not None:
            limit = min(len(wfn), limit)
        return solve_variational(op, min(args.nroots, len(wfn)), c0, max_subspace=limit, **solver_options)

    def product_stats():
        if args.sci_stats:
            if args.direct:
                print(f'direct {op.product_info}')
                print(f'product {op.last_product_stats}')
            else:
                print(f"matrix {getattr(backend, 'last_matrix_stats', {})}")
    energies, coeffs = diagonalize()
    products = f'direct {args.direct_strategy}' if args.direct else 'stored CSR'
    print(f'{args.backend} backend, {products}, Davidson: N={nelec}, 2S={args.two_s}, orbitals={ham.nbasis}, full dimension={dimension}')
    print(f'iteration=0 ncsf={len(wfn)} energies={energies}')
    product_stats()
    if not args.full:
        for iteration in count(1):
            if args.max_iterations is not None and iteration > args.max_iterations:
                raise RuntimeError('HCI did not stop within max-iterations')
            added, stats = scsf.add_csf_hci(ham, wfn, np.max(np.abs(coeffs), axis=1), eps=args.eps, backend=backend, max_csfs=args.max_csfs, max_candidates=args.max_candidates, max_configurations=args.max_configurations, selection_rule=args.selection_rule, parent_fraction=args.parent_fraction, implementation=args.selection, screening=not args.no_screening, return_stats=True)
            if args.sci_stats:
                print(f'selection {stats}')
            if not added:
                if len(energies) < args.nroots:
                    raise RuntimeError('selection stopped before reaching nroots; choose a larger initial basis or --full')
                break
            op.update(ham, wfn)
            guess = np.pad(coeffs, ((0, added), (0, 0)))
            energies, coeffs = diagonalize(guess)
            print(f'iteration={iteration} added={added} ncsf={len(wfn)} energies={energies}')
            product_stats()
    for root, energy in enumerate(energies):
        print(f'root={root} energy={energy:.12f} hartree')
if __name__ == '__main__':
    main()
