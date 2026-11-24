# SCSF; licensed under GPL-3.0-or-later.
"""Spin-adapted CI and selected CI with native Hamiltonian products.

Examples (run from the repository with SCSF importable)::

    python -m scsf --integrals scsf/test/data/lih_sto3g.npz --eps 1e-4
    python -m scsf --integrals scsf/test/data/lih_sto3g.npz --natorb sci
    python -m scsf --xyz examples/water.xyz --basis sto-3g --eps 1e-4
    python -m scsf --xyz examples/water.xyz --basis cc-pvdz --ncore 1 --kernel nminus2

PySCF is required for the XYZ input route.
"""
import argparse
from collections import Counter
from itertools import count
import re
from time import perf_counter

import numpy as np
import scsf
from ..config import default_cli_threads
from ..csf import _configuration
from ..integrals import rotate_hamiltonian
from .molecular import active_mo_symmetry, correlated_density, integrals_from_scf, run_scf
from .solvers import solve_ci


def memory_bytes(value):
    match = re.fullmatch(r'([0-9]+)\s*(B|KiB|MiB|GiB|TiB)?', value.strip(), re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError('expected bytes or an integer with a KiB, MiB, GiB or TiB suffix')
    number, unit = match.groups()
    power = ('b', 'kib', 'mib', 'gib', 'tib').index((unit or 'B').lower())
    size = int(number)*1024**power
    if size >= 1 << (np.dtype(np.uintp).itemsize*8):
        raise argparse.ArgumentTypeError('memory limit exceeds the native byte count range')
    return size


def integrals(args, parser, *, packed_output=False):
    if args.integrals:
        if args.natorb in ('mp2', 'ccsd'):
            parser.error(f'--natorb {args.natorb} requires --xyz')
        if args.ncore:
            parser.error('--ncore applies to XYZ input; snapshots must already contain active integrals')
        if args.point_group is not None:
            parser.error('--point-group applies to XYZ input. Snapshots supply their own orbital irreps')
        try:
            ham, nelec = scsf.load_hamiltonian(args.integrals)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        nelec = args.nelec if args.nelec is not None else nelec
        if nelec is None:
            parser.error('--nelec is required if the array snapshot has no nelec field')
        return ham, nelec
    if not args.basis:
        parser.error('--basis is required with --xyz')
    if args.nelec is not None:
        parser.error('--nelec applies to array snapshot input; XYZ uses the molecular charge')
    group = args.point_group if args.target_irrep is None else args.point_group or 'auto'
    try:
        mf = run_scf(args.xyz, args.basis, args.charge, args.two_s, args.ncore, parser,
                     point_group=group)
    except ValueError as exc:
        parser.error(str(exc))
    mo_coeff = None
    if args.natorb in ('mp2', 'ccsd'):
        method = args.natorb.upper()
        print(f'Generating natural orbitals from {method}', flush=True)
        symmetry = None if group is None else active_mo_symmetry(mf, args.ncore)
        rotation = natural_orbitals(correlated_density(mf, args.ncore, args.natorb), method,
                                    symmetry=symmetry)
        mo_coeff = mf.mo_coeff.copy()
        mo_coeff[:, args.ncore:] = mo_coeff[:, args.ncore:] @ rotation
    ham, nelec, _ = integrals_from_scf(mf, args.ncore, mo_coeff=mo_coeff, packed=packed_output,
                                      point_group=group is not None)
    return ham, nelec


def natural_orbitals(density, method, *, symmetry=None):
    if not np.all(np.isfinite(density)):
        raise RuntimeError(f'{method} produced a nonfinite 1-RDM')
    symmetric = (density + density.T) * .5
    if symmetry is None:
        occupations, orbitals = np.linalg.eigh(symmetric)
    else:
        labels = np.asarray(symmetry.orbital_irreps, dtype=np.intp)
        if np.any(np.abs(symmetric[labels[:, None] != labels]) > 1e-10):
            raise ValueError(f'{method} 1-RDM mixes orbital irreps')
        occupations = np.empty(len(labels))
        orbitals = np.zeros_like(symmetric)
        for code in range(len(symmetry.names)):
            ids = np.flatnonzero(labels == code)
            if len(ids):
                block = np.ix_(ids, ids)
                occupations[ids], orbitals[block] = np.linalg.eigh(symmetric[block])
    order = np.argsort(-occupations, kind='stable')
    values = ' '.join(f'{occupation:.8f}' for occupation in occupations[order])
    print(f'{method} natural occupations=[{values}]', flush=True)
    return orbitals[:, order]

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
    space.add_argument('--point-group', nargs='?', const='auto',
                       help='enable orbital symmetry for XYZ input, optionally with a subgroup')
    space.add_argument('--target-irrep', help='restrict the CI to this irrep (name or numeric code)')
    space.add_argument('--reference', help='initial CSF path, right-padded with zeros (e.g. 2222 or 22ud)')
    space.add_argument('--natorb', nargs='?', const='sci', type=str.lower, choices=('sci', 'mp2', 'ccsd'),
                       help='use natural orbitals from a pilot SCI run (sci) or XYZ with PySCF (mp2, ccsd); '
                            'bare --natorb selects sci')
    davidson.add_argument('--nroots', type=int, default=1)
    selection.add_argument('--eps', type=float, default=0.0001, help='HCI selection threshold')
    selection.add_argument('--stop-growth-fraction', type=float, default=0.01,
                           help='solve and stop after an append adding at most this fraction of current CSFs '
                                '(default: 0.01; use 0 for strict HCI)')
    selection.add_argument('--max-iterations', type=int, help='optional SCI iteration cap; default: unlimited')
    products.add_argument('--kernel', choices=scsf.csf_structured_op.kernels,
                          default='nminus2', help='Hamiltonian kernel (default: nminus2)')
    selection.add_argument('--selection-unit', choices=('csf', 'configuration'), default='csf',
                           help='HCI selection unit (default: csf)')
    parser.add_argument('--quiet', action='store_true', help='show cycle summaries and final timings')
    parser.add_argument('--sci-stats', action='store_true', help='show detailed phase timings and totals')
    products.add_argument('--nthreads', type=int, default=default_cli_threads(),
                          help='native workers (default: available physical cores, up to 256)')
    products.add_argument('--max-mem', type=memory_bytes,
                          help='shared budget for native HCI tables, schedules, coefficients, plans and product arrays '
                               '(bytes or a size such as 4GiB, default: unlimited)')
    products.add_argument('--structured-pair-accumulation', choices=('gather','private'),
                        help='N−2 pair accumulation: gather auxiliary amplitudes or stream into private outputs')
    products.add_argument('--worker-cpus', type=int, nargs='+',
                          help='pin workers to distinct allowed Linux CPU IDs, one per --nthreads worker')
    davidson.add_argument('--davidson-jd-iterations', type=int, default=0,
                        help='maximum MINRES steps per projected Jacobi–Davidson correction')
    davidson.add_argument('--davidson-block-size', type=int, default=0, help='size of the exact leading-CSF preconditioner block')
    davidson.add_argument('--davidson-tol', type=float, default=1e-6)
    davidson.add_argument('--davidson-maxiter', type=int, default=200)
    davidson.add_argument('--davidson-max-subspace', type=int)
    return parser


def configure_driver(args, parser):
    if not 1 <= args.nthreads <= 256:
        parser.error('--nthreads must be in 1..256')
    if args.structured_pair_accumulation is not None and args.kernel != 'nminus2':
        parser.error('--structured-pair-accumulation requires --kernel nminus2')
    if args.nroots < 1 or (args.max_iterations is not None and args.max_iterations < 1) or not np.isfinite(args.eps) or args.eps < 0:
        parser.error('require nroots/max-iterations >= 1 and finite eps >= 0')
    if not 0. <= args.stop_growth_fraction <= 1.:
        parser.error('--stop-growth-fraction must be in [0, 1]')
    if min(args.davidson_block_size, args.davidson_jd_iterations) < 0:
        parser.error('Davidson block size and JD iterations must be nonnegative')


def initial_space(args, parser, ham, nelec):
    try:
        target = args.target_irrep
        if target is not None and target.isdecimal():
            target = int(target)
        wfn = scsf.csf_wfn(ham.nbasis, nelec, args.two_s, symmetry=ham.symmetry, target_irrep=target)
        dimension = scsf.csf_dimension(ham.nbasis, nelec, args.two_s, symmetry=ham.symmetry,
                                       target_irrep=target)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.nroots > dimension:
        parser.error('nroots exceeds the target-spin dimension')
    reference = args.reference
    if reference is None:
        pairs = (nelec - args.two_s) // 2
        reference = '2' * pairs + 'u' * args.two_s
    reference = reference.ljust(ham.nbasis, '0')
    try:
        if args.reference is None and wfn.target_irrep is not None:
            reference = wfn._reference_in_sector(_configuration(reference))
        if args.selection_unit == 'configuration':
            label = wfn._validate(reference)
            wfn.add_configuration(_configuration(label))
        else:
            wfn.add_csf(reference)
        if wfn.target_irrep is not None and len(wfn) < args.nroots:
            wfn._fill_sector_seed(args.nroots, complete=args.selection_unit == 'configuration')
    except ValueError as exc:
        parser.error(f'invalid --reference {reference!r} ({exc})')
    return wfn, dimension


def make_operator(args, ham, wfn):
    return scsf.csf_structured_op(
        ham, wfn, strategy=args.kernel, nthreads=args.nthreads, worker_cpus=args.worker_cpus,
        max_mem=args.max_mem, pair_accumulation=args.structured_pair_accumulation)


def diagonalize(args, op, c0=None, *, measurements=None, progress=False):
    roots, size = min(args.nroots, op.shape[0]), op.shape[0]
    limit = args.davidson_max_subspace
    if limit is not None:
        limit = min(size, limit)
    start = perf_counter()
    diagonal = op.native_diagonal()
    diagonal_seconds = perf_counter()-start
    preconditioner_block = None
    preconditioner_memory_peak = 0
    block_start = perf_counter()
    if args.davidson_block_size:
        rows = np.arange(size)
        importance = np.zeros(size) if c0 is None else np.max(np.abs(c0), axis=1)
        rows = np.lexsort((rows, diagonal, -importance))[:args.davidson_block_size]
        basis = op.basis_context
        labels = op.native_labels(rows)
        sectors = {}
        for i, label in enumerate(labels):
            code = None if op.symmetry is None else op.symmetry.configuration_irrep(label)
            sectors.setdefault(code, []).append(i)
        matrix = np.zeros((len(rows), len(rows)))
        for code, indices in sectors.items():
            subspace = scsf.csf_wfn(basis['nbasis'], basis['nelec'], basis['two_s'],
                                   symmetry=op.symmetry, target_irrep=code)
            for i in indices:
                subspace.add_csf(labels[i])
            block_op = op.with_basis(subspace, concurrent=True)
            matrix[np.ix_(indices, indices)] = block_op.matmat(np.eye(len(indices)))
            preconditioner_memory_peak = max(preconditioner_memory_peak,
                op.product_info['tracked_memory_bytes'] + block_op.product_info['peak_tracked_memory_bytes'])
            del block_op
        preconditioner_block = rows, matrix
    block_seconds = perf_counter()-block_start
    stats = {}
    measure_products = measurements is not None or args.sci_stats
    products = {}

    def measured_product(vectors):
        start = perf_counter()
        result = op.native_matmat(vectors)
        elapsed = perf_counter()-start
        width = vectors.shape[1]
        phase = 'solver' if products else 'first'
        key = f'{phase}/{width}'
        bucket = products.setdefault(key, {'calls': 0, 'seconds': 0., 'native': {}})
        bucket['calls'] += 1
        bucket['seconds'] += elapsed
        for name, value in op.last_product_stats.items():
            if name.endswith('_ns') or name in ('plan_hits', 'plans_built'):
                native = bucket['native']
                native[name] = native.get(name, 0) + value
        return result

    product = measured_product if measure_products else op.native_matmat
    energies, vectors, converged = solve_ci(product, diagonal, roots=roots, c0=c0,
        tol=args.davidson_tol, maxiter=args.davidson_maxiter, max_subspace=limit, stats=stats, verbose=progress,
        preconditioner_block=preconditioner_block, jd_iterations=args.davidson_jd_iterations)
    residual = np.asarray(stats['residual_norms'])
    if not converged:
        raise RuntimeError(f'Davidson did not converge, Ritz residuals {residual}')
    if progress:
        solve_seconds = stats['total_ns']*1e-9
        print(f'  Davidson converged in {solve_seconds:.3f} s, diagonal {diagonal_seconds:.3f} s, '
              f'preconditioner {block_seconds:.3f} s, max Ritz residual={np.max(residual):.3e} Eh', flush=True)
    if measurements is not None:
        measurements.update(solver=stats, products=products, diagonal_seconds=diagonal_seconds,
                            preconditioner_seconds=block_seconds,
                            residuals=residual.tolist(), preconditioner_memory_peak=preconditioner_memory_peak)
    if args.sci_stats:
        print(f'cycle setup={op.setup_seconds:.6f} diagonal={diagonal_seconds:.6f} '
              f'preconditioner={block_seconds:.6f} product={stats["product_ns"]*1e-9:.6f} '
              f'product_calls={stats["product_calls"]} product_columns={stats["product_columns"]} '
              f'jd_product_calls={stats["jd_product_calls"]} '
              f'davidson={(stats["total_ns"]-stats["product_ns"])*1e-9:.6f} '
              f'residuals={residual}')
        print(f'solver {stats}')
        print(f'products_by_phase_width {products}')
    return energies+op.ecore, vectors


def grow_guess(vectors, size, remap=None):
    """Extend solver columns after an append or native row remap."""
    if remap is None or not remap.size:
        return np.pad(vectors, ((0, size-len(vectors)), (0, 0)))
    result = np.zeros((size, vectors.shape[1]), dtype=vectors.dtype)
    result[remap] = vectors
    return result


def print_timings(timing, wall):
    rows = [('Integrals', timing['integrals'])]
    if 'natural_orbitals' in timing:
        rows.append(('SCI density and orbital rotation', timing['natural_orbitals']))
    rows += [
        ('Initial operator setup', timing['initial_setup']),
        ('Hamiltonian diagonal', timing['diagonal']),
        ('Davidson preconditioner setup', timing['preconditioner']),
        ('Hamiltonian products inside Davidson', timing['solver_products']),
        ('Ritz projection and solve', timing['ritz']),
        ('Davidson orthogonalization', timing['orthogonalization']),
        ('Davidson restarts', timing['restart']),
        ('Davidson basis storage', timing['basis_append']),
        ('Other Davidson work', timing['solver_bookkeeping']),
        ('HCI selection', timing['selection']),
        ('Basis extension', timing['basis_extend']),
        ('Operator replacement', timing['replacement']),
        ('Diagonal reuse', timing['diagonal_reuse']),
        ('Selection bookkeeping and guess remap', timing['selection_overhead'] + timing['guess_remap']),
    ]
    rows += [('Other driver overhead', max(0., wall-sum(seconds for _, seconds in rows))),
             ('Total', wall)]
    width = max(len(label) for label, _ in rows)
    print(f'\n{"Phase":<{width}}  {"Wall (s)":>10}  {"Share":>7}')
    for label, seconds in rows:
        print(f'{label:<{width}}  {seconds:10.3f}  {seconds/wall:7.1%}')


def main(argv=None):
    started = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_driver(args, parser)
    try:
        run_hci(args, parser, started)
    except MemoryError as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')


def run_hci(args, parser, started):
    timing = Counter()
    if not args.quiet:
        print(f'Loading integrals from {args.integrals or args.xyz}', flush=True)
    start = perf_counter()
    ham, nelec = integrals(args, parser, packed_output=True)
    timing['integrals'] = perf_counter()-start

    pilot_peak = 0
    if args.natorb == 'sci':
        print('Pilot SCI run for natural orbitals', flush=True)
        _, vectors, op, pilot_peak = hci_pass(args, parser, ham, nelec, timing, started=started)
        start = perf_counter()
        density = np.zeros((ham.nbasis, ham.nbasis))
        for root in range(vectors.shape[1]):
            density += op.native_rdm1(vectors[:, root])
        pilot_peak = max(pilot_peak, op.product_info['peak_tracked_memory_bytes'])
        del vectors, op
        rotation = natural_orbitals(density / args.nroots, 'SCI', symmetry=ham.symmetry)
        ham = rotate_hamiltonian(ham, rotation)
        timing['natural_orbitals'] = perf_counter()-start
    if args.natorb is not None:
        print('HCI run in natural orbitals', flush=True)
    energies, _, _, peak_memory = hci_pass(
        args, parser, ham, nelec, timing, started=None if args.natorb == 'sci' else started)
    for root, energy in enumerate(energies):
        print(f'root={root} energy={energy:.12f} hartree')
    wall = perf_counter()-started
    print_timings(timing, wall)
    print(f'Tracked native peak memory {max(peak_memory, pilot_peak)/1024**2:.2f} MiB')
    if args.sci_stats:
        print('timing_seconds ' + ' '.join(f'{phase}={seconds:.6f}' for phase, seconds in timing.items()))
        print(f'driver_wall_seconds={wall:.6f}')


def hci_pass(args, parser, ham, nelec, timing, *, started=None):
    progress = not args.quiet
    preconditioner_memory_peak = 0
    start = perf_counter()
    wfn, dimension = initial_space(args, parser, ham, nelec)
    op = make_operator(args, ham, wfn)
    timing['initial_setup'] += perf_counter()-start
    if args.sci_stats and started is not None:
        print(f'driver_startup_seconds={perf_counter()-started:.6f} cpus={args.worker_cpus}')
    print(f'{op.strategy}, {args.selection_unit} HCI selection, Davidson: '
          f'N={nelec}, 2S={args.two_s}, orbitals={ham.nbasis}, full dimension={dimension}')
    if ham.symmetry is not None:
        represented = sorted({ham.symmetry.configuration_irrep(cfg) for cfg in wfn._configs})
        names = ','.join(ham.symmetry.names[g] for g in represented)
        print(f'Point group {ham.symmetry.group}, seed irreps {names}')

    def solve_step(iteration, added=0, guess=None, *, final=False):
        nonlocal preconditioner_memory_peak
        if progress:
            stage = 'Final Davidson solve' if final else 'Davidson solve'
            print(f'HCI step {iteration} | {stage} for {op.shape[0]:,} CSFs '
                  f'(residual tolerance {args.davidson_tol:g} Eh)', flush=True)
        measurements = {}
        energies, vectors = diagonalize(args, op, guess, measurements=measurements, progress=progress)
        for phase in ('diagonal', 'preconditioner'):
            timing[phase] += measurements[f'{phase}_seconds']
        solver = measurements['solver']
        preconditioner_memory_peak = max(preconditioner_memory_peak, measurements['preconditioner_memory_peak'])
        for phase in ('ritz', 'orthogonalization', 'restart', 'basis_append'):
            timing[phase] += solver[f'{phase}_ns']*1e-9
        timing['solver_products'] += solver['product_ns']*1e-9
        timing['solver_other'] += (solver['total_ns']-solver['product_ns'])*1e-9
        timing['solver_bookkeeping'] += solver['other_ns']*1e-9
        addition = f' added={added}' if added else ''
        values = ' '.join(f'{energy:.12f}' for energy in energies)
        print(f'iteration={iteration}{addition} ncsf={op.shape[0]} energies=[{values}]', flush=True)
        if args.sci_stats:
            print(f'kernel {op.product_info}')
            print(f'product {op.last_product_stats}')
        return energies, vectors

    energies, vectors = solve_step(0)
    for iteration in count(1):
        if args.max_iterations is not None and iteration>args.max_iterations:
            raise RuntimeError('HCI did not stop within max-iterations')
        old_size = op.shape[0]
        if progress:
            print(f'HCI step {iteration} | Selecting from {old_size:,} CSFs at eps={args.eps:g}', flush=True)
        start = perf_counter()
        weights = np.max(np.abs(vectors), axis=1)
        weights_seconds = perf_counter()-start
        start = perf_counter()
        remap, stats = op.select_hci(weights, args.eps, selection_unit=args.selection_unit)
        selection_seconds = perf_counter()-start
        breakdown = {phase: stats.get(f'{phase}_ns', 0)*1e-9
                     for phase in ('selection', 'basis_extend', 'replacement', 'diagonal_reuse')}
        timing.update(breakdown)
        timing['weights'] += weights_seconds
        timing['selection_update'] += selection_seconds
        timing['selection_overhead'] += weights_seconds + selection_seconds - sum(breakdown.values())
        if args.sci_stats:
            print(f'selection {stats}')
            print(f'growth weights={weights_seconds:.6f} selection_and_update={selection_seconds:.6f}')
        added = stats['added']
        if not added:
            if len(energies)<args.nroots:
                raise RuntimeError('selection stopped before reaching nroots. Lower --eps or choose another --reference.')
            if progress:
                print(f'HCI converged with zero additions ({selection_seconds:.3f} s)', flush=True)
            break
        fraction = added/old_size
        final = fraction <= args.stop_growth_fraction and op.shape[0] >= args.nroots
        if progress:
            print(f'  Added {added:,} CSFs ({fraction:.3%}), now {op.shape[0]:,} '
                  f'({selection_seconds:.3f} s for selection and update)', flush=True)
        if final:
            print(f'HCI growth cutoff reached ({fraction:.3%} ≤ {args.stop_growth_fraction:.3%}). '
                  'Solving the enlarged space and stopping.', flush=True)
        start = perf_counter()
        guess = grow_guess(vectors, op.shape[0], remap)
        remap_seconds = perf_counter()-start
        timing['guess_remap'] += remap_seconds
        if args.sci_stats:
            print(f'guess remap={remap_seconds:.6f}')
        energies, vectors = solve_step(iteration, added, guess, final=final)
        if final:
            break
    peak_memory = max(op.product_info['peak_tracked_memory_bytes'], preconditioner_memory_peak)
    return energies, vectors, op, peak_memory


if __name__ == '__main__':
    main()
