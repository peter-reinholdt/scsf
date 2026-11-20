# SCSF; licensed under GPL-3.0-or-later.
"""Spin-adapted CI and selected CI with native Hamiltonian products.

Examples (run from the repository with SCSF importable)::

    python -m scsf --integrals scsf/test/data/lih_sto3g.npz --eps 1e-4
    python -m scsf --xyz examples/water.xyz --basis sto-3g --eps 1e-4
    python -m scsf --xyz examples/water.xyz --basis cc-pvdz --ncore 1 --kernel nminus2

PySCF is required for the XYZ input route.
"""
import argparse
from itertools import count
from time import perf_counter

import numpy as np
import scsf
from ..config import default_cli_threads
from ..csf import _configuration
from .molecular import integrals_from_scf, run_scf
from .solvers import solve_ci


def integrals(args, parser):
    if args.integrals:
        if args.ncore:
            parser.error('--ncore applies to XYZ input; snapshots must already contain active integrals')
        with np.load(args.integrals, allow_pickle=False) as data:
            nelec = args.nelec if args.nelec is not None else int(data['nelec']) if 'nelec' in data else None
            if nelec is None:
                parser.error('--nelec is required if the array snapshot has no nelec field')
            ecore, one = float(data['ecore']), data['one']
            if 'two' in data:
                return scsf.Hamiltonian(ecore, one, data['two']), nelec
            if 'packed_eri' in data:
                return scsf.PackedHamiltonian(ecore, one, data['packed_eri']), nelec
            if 'cholesky_factors' in data:
                return scsf.CholeskyHamiltonian(ecore, one, data['cholesky_factors']), nelec
            if 'thc_collocation' in data and 'thc_core' in data:
                return scsf.THCHamiltonian(ecore, one, data['thc_collocation'], data['thc_core']), nelec
            parser.error('snapshot needs two, packed_eri, cholesky_factors, or thc_collocation and thc_core')
    if not args.basis:
        parser.error('--basis is required with --xyz')
    if args.nelec is not None:
        parser.error('--nelec applies to array snapshot input; XYZ uses the molecular charge')
    mf = run_scf(args.xyz, args.basis, args.charge, args.two_s, args.ncore, parser)
    ham, nelec, _ = integrals_from_scf(mf, args.ncore)
    return (ham, nelec)

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    space = parser.add_argument_group('Integrals and orbital space')
    selection = parser.add_argument_group('Selection and growth')
    products = parser.add_argument_group('Hamiltonian products')
    davidson = parser.add_argument_group('Davidson solver')
    source = space.add_mutually_exclusive_group(required=True)
    source.add_argument('--integrals')
    source.add_argument('--xyz')
    space.add_argument('--basis')
    space.add_argument('--charge', type=int, default=0)
    space.add_argument('--ncore', type=int, default=0)
    space.add_argument('--nelec', type=int)
    space.add_argument('--two-s', type=int, default=0, help='twice the target total spin')
    space.add_argument('--reference', help='initial CSF path, right-padded with zeros (e.g. 2222 or 22ud)')
    davidson.add_argument('--nroots', type=int, default=1)
    selection.add_argument('--eps', type=float, default=0.0001, help='HCI selection threshold')
    selection.add_argument('--max-iterations', type=int, help='optional SCI iteration cap; default: unlimited')
    products.add_argument('--kernel', choices=scsf.csf_structured_op.kernels,
                          default='nminus2', help='Hamiltonian kernel (default: nminus2)')
    selection.add_argument('--selection-unit', choices=('csf', 'configuration'), default='csf',
                           help='HCI selection unit (default: csf)')
    selection.add_argument('--no-screening', action='store_true', help='disable HCI transfer screening')
    selection.add_argument('--max-ordered-bytes', type=int, default=16*1024**2,
                           help='memory ceiling for ordered HCI transfer bounds')
    parser.add_argument('--sci-stats', action='store_true')
    products.add_argument('--nthreads', type=int, default=default_cli_threads(),
                          help='native workers (default: available physical cores, up to 256)')
    products.add_argument('--max-product-scratch-bytes', type=int, default=2*1024**3,
                          help='product array ceiling (default: 2 GiB)')
    products.add_argument('--max-plan-bytes', type=int, help='CIH plan or particle–hole neighbor cache cap')
    products.add_argument('--max-structured-metadata-bytes', type=int, default=8*1024**3)
    products.add_argument('--max-structured-coefficient-bytes', type=int, default=256*1024**2)
    products.add_argument('--structured-pair-accumulation', choices=('gather','private'),
                        help='N−2 pair accumulation: gather auxiliary amplitudes or stream into private outputs')
    products.add_argument('--worker-cpus', type=int, nargs='+',
                          help='pin workers to distinct allowed Linux CPU IDs, one per --nthreads worker')
    davidson.add_argument('--davidson-jd-iterations', type=int, default=0,
                        help='maximum MINRES steps per projected Jacobi–Davidson correction')
    davidson.add_argument('--davidson-block-size', type=int, default=0, help='size of the exact leading-CSF preconditioner block')
    davidson.add_argument('--davidson-tol', type=float, default=1e-10)
    davidson.add_argument('--davidson-maxiter', type=int, default=200)
    davidson.add_argument('--davidson-max-subspace', type=int)
    return parser


def configure_driver(args, parser):
    if not 1 <= args.nthreads <= 256:
        parser.error('--nthreads must be in 1..256')
    if args.structured_pair_accumulation is not None and args.kernel != 'nminus2':
        parser.error('--structured-pair-accumulation requires --kernel nminus2')
    budgets = (args.max_product_scratch_bytes, args.max_structured_metadata_bytes,
               args.max_structured_coefficient_bytes, args.max_ordered_bytes)
    if min(budgets) < 0 or (args.max_plan_bytes is not None and args.max_plan_bytes < 0):
        parser.error('memory limits must be nonnegative')
    if args.nroots < 1 or (args.max_iterations is not None and args.max_iterations < 1) or not np.isfinite(args.eps) or args.eps < 0:
        parser.error('require nroots/max-iterations >= 1 and finite eps >= 0')
    if min(args.davidson_block_size, args.davidson_jd_iterations) < 0:
        parser.error('Davidson block size and JD iterations must be nonnegative')


def initial_space(args, parser, ham, nelec):
    try:
        wfn = scsf.csf_wfn(ham.nbasis, nelec, args.two_s)
        dimension = scsf.csf_dimension(ham.nbasis, nelec, args.two_s)
    except ValueError as exc:
        parser.error(str(exc))
    if args.nroots > dimension:
        parser.error('nroots exceeds the target-spin dimension')
    reference = args.reference
    if reference is None:
        pairs = (nelec - args.two_s) // 2
        reference = '2' * pairs + 'u' * args.two_s
    reference = reference.ljust(ham.nbasis, '0')
    try:
        if args.selection_unit == 'configuration':
            label = wfn._validate(reference)
            wfn.add_configuration(_configuration(label))
        else:
            wfn.add_csf(reference)
    except ValueError as exc:
        parser.error(f'invalid --reference {reference!r} ({exc})')
    return wfn, dimension


def make_operator(args, ham, wfn):
    return scsf.csf_structured_op(
        ham, wfn, strategy=args.kernel, nthreads=args.nthreads, worker_cpus=args.worker_cpus,
        max_metadata_bytes=args.max_structured_metadata_bytes,
        max_coefficient_bytes=args.max_structured_coefficient_bytes,
        max_product_scratch_bytes=args.max_product_scratch_bytes,
        pair_accumulation=args.structured_pair_accumulation,
        max_plan_bytes=args.max_plan_bytes, max_ordered_bytes=args.max_ordered_bytes)


def diagonalize(args, op, c0=None):
    roots, size = min(args.nroots, op.shape[0]), op.shape[0]
    limit = args.davidson_max_subspace
    if limit is not None:
        limit = min(size, limit)
    start = perf_counter()
    diagonal = op.native_diagonal()
    diagonal_seconds = perf_counter()-start
    preconditioner_block = None
    block_start = perf_counter()
    if args.davidson_block_size:
        rows = np.arange(size)
        importance = np.zeros(size) if c0 is None else np.max(np.abs(c0), axis=1)
        rows = np.lexsort((rows, diagonal, -importance))[:args.davidson_block_size]
        basis = op.basis_context
        subspace = scsf.csf_wfn(basis['nbasis'], basis['nelec'], basis['two_s'])
        for label in op.native_labels(rows):
            subspace.add_csf(label)
        preconditioner_block = rows, op.with_basis(subspace).matmat(np.eye(len(rows)))
    block_seconds = perf_counter()-block_start
    stats = {} if args.sci_stats else None
    energies, vectors, converged = solve_ci(op.native_matmat, diagonal, roots=roots, c0=c0,
        tol=args.davidson_tol, maxiter=args.davidson_maxiter, max_subspace=limit, stats=stats,
        preconditioner_block=preconditioner_block, jd_iterations=args.davidson_jd_iterations)
    residual = np.linalg.norm(op.native_matmat(vectors)-vectors*energies, axis=0)
    if not converged or np.any(residual > args.davidson_tol):
        raise RuntimeError(f'Davidson did not converge, exact residuals {residual}')
    if stats is not None:
        print(f'cycle setup={op.setup_seconds:.6f} diagonal={diagonal_seconds:.6f} '
              f'preconditioner={block_seconds:.6f} product={stats["product_ns"]*1e-9:.6f} '
              f'product_calls={stats["product_calls"]} product_columns={stats["product_columns"]} '
              f'jd_product_calls={stats["jd_product_calls"]} '
              f'davidson={(stats["total_ns"]-stats["product_ns"])*1e-9:.6f} residuals={residual}')
    return energies+op.ecore, vectors


def grow_guess(vectors, size, remap=None):
    """Extend solver columns after an append or native row remap."""
    if remap is None or not remap.size:
        return np.pad(vectors, ((0, size-len(vectors)), (0, 0)))
    result = np.zeros((size, vectors.shape[1]), dtype=vectors.dtype)
    result[remap] = vectors
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_driver(args, parser)
    ham, nelec = integrals(args, parser)
    wfn, dimension = initial_space(args, parser, ham, nelec)
    op = make_operator(args, ham, wfn)

    def product_stats():
        if args.sci_stats:
            print(f'kernel {op.product_info}')
            print(f'product {op.last_product_stats}')

    energies, vectors = diagonalize(args, op)
    print(f'{op.strategy}, {args.selection_unit} HCI selection, Davidson: '
          f'N={nelec}, 2S={args.two_s}, orbitals={ham.nbasis}, full dimension={dimension}')
    print(f'iteration=0 ncsf={op.shape[0]} energies={energies}')
    product_stats()
    for iteration in count(1):
        if args.max_iterations is not None and iteration>args.max_iterations:
            raise RuntimeError('HCI did not stop within max-iterations')
        weights = np.max(np.abs(vectors), axis=1)
        remap, stats = op.select_hci(weights, args.eps, selection_unit=args.selection_unit,
                                    screening=not args.no_screening)
        if args.sci_stats:
            print(f'selection {stats}')
        added = stats['added']
        if not added:
            if len(energies)<args.nroots:
                raise RuntimeError('selection stopped before reaching nroots. Lower --eps or choose another --reference.')
            break
        guess = grow_guess(vectors, op.shape[0], remap)
        energies, vectors = diagonalize(args, op, guess)
        print(f'iteration={iteration} added={added} ncsf={op.shape[0]} energies={energies}')
        product_stats()
    for root, energy in enumerate(energies):
        print(f'root={root} energy={energy:.12f} hartree')


if __name__ == '__main__':
    main()
