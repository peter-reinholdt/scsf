"""Direct and bounded semi-direct GUGA Hamiltonian products.

The stored native CSR remains the production default. Contribution screening
is nonlinear; disable all screening for exact linear products and residual checks.
"""
import numpy as np
import os
from time import perf_counter_ns

from .csf import CSFSpinBackend, csf_wfn, _integer, _real_vector


class csf_direct_op:
    """Electronic Hamiltonian products in an arbitrary selected CSF space.

    ``compact-spin-guga`` is the molecular default: ranked connections, exact
    occupation masks, bounded spin summaries and a 16-byte coefficient memo.
    ``compact-cache-guga`` adds aligned cache buckets and paired encoding.
    ``compact-adaptive-guga`` retains workers/workspace and uses active-source
    traversal for sparse inputs; it is useful on large multicore water jobs.

    ``wide-spin-guga`` uses 24-byte keys supporting more open-shell sites.
    ``tuga`` caches integral-independent selected coefficient tiles, improving
    repeated spin-rich products. ``selected-guga`` shares selected-path trie
    recurrences; ``hybrid-guga`` uses those recurrences only outside the tiny
    memo format. ``weighted-guga`` contracts vector-weighted trie environments,
    trading runtime for small temporary and persistent storage when scalar
    memoization is disabled. All have exact scalar fallbacks for cache/work
    limits, unsupported path lengths and incomplete selected spin manifolds.

    ``semi-direct`` eagerly caches complete numerical occupation rows within
    ``max_hamiltonian_bytes``. Include construction in solve timings. Other
    strategies store no off-diagonal Hamiltonian values. Graph/path/metadata,
    coefficient-tile and selected-trie allocations have separate byte caps.
    ``max_product_scratch_bytes`` bounds replicated output and omission arrays
    (default 256 MiB), reducing active workers and then tiling RHS columns.
    Input/output, coefficient metadata, memo and coupling workspaces have
    separate storage; this is not a total-process memory limit. A cap smaller
    than one column raises MemoryError before allocating product scratch.
    ``product_accumulation='gather'`` is an experimental exact compact mode
    with disjoint output rows and no replicated output, at the cost of directed
    coupling evaluation. ``spin_encoder='portable'|'bmi2'|'auto'`` changes only
    encoding, preserving memo layout. Optional ``worker_cpus`` pins adaptive
    workers to distinct allowed Linux CPU IDs; caller affinity is restored.
    ``compact_spin_kernel='short'`` experimentally specializes one/two-generator
    rank-two scalar contractions without changing memo capacity or key layout.
    ``basis_storage='packed'`` and ``spin_storage='checkpoints'`` optionally
    trade reconstruction work for lower basis/path storage.

    Diagonals are exact. Off-diagonal contributions with ``abs(H_ij*c_j)``
    below ``product_tol`` are omitted; ``matrix_cutoff`` independently screens
    matrix entries. Prepared strategies additionally support bounded generator
    screening using ``generator_tol`` and ``generator_screen='budget'|'term'``.
    ``last_error_inf`` and ``last_error_l2`` certify omitted terms per column,
    excluding floating-point roundoff. Screened products need not be linear.
    For exact products, set every tolerance to zero or use ``unscreened=True``
    with a prepared strategy. Reset the solver history when changing screening.
    """
    _strategies = ('compact-spin-guga', 'compact-cache-guga', 'compact-adaptive-guga',
                   'wide-spin-guga', 'semi-direct', 'tuga', 'selected-guga',
                   'hybrid-guga', 'weighted-guga')
    _prepared_strategies = frozenset(_strategies) - {'wide-spin-guga', 'semi-direct'}

    def __init__(self, ham, wfn, *, backend=None, strategy='compact-spin-guga',
                 product_tol=1e-12, matrix_cutoff=0., nthreads=None,
                 max_graph_bytes=64*1024*1024, max_path_bytes=256*1024*1024,
                 max_ordered_bytes=16*1024*1024, max_hamiltonian_bytes=128*1024*1024,
                 profile_couplings=False, max_batch_elements=8192, batch_min_pairs=16, batch_min_terms=0, collect_spin_stats=True,
                 generator_tol=0., generator_screen='budget', basis_storage='labels', spin_storage='full',
                 max_symbolic_bytes=16*1024*1024, max_trie_bytes=16*1024*1024,
                 max_product_scratch_bytes=256*1024*1024,
                 product_accumulation='private', spin_encoder='auto', worker_cpus=None,
                 compact_spin_kernel='general'):
        if not isinstance(wfn, csf_wfn):
            raise TypeError('csf_direct_op requires csf_wfn')
        self.backend = backend if backend is not None else CSFSpinBackend(ham, matrix_index='zobrist')
        if (not isinstance(self.backend, CSFSpinBackend) or
                self.backend.coupling_engine != 'guga'):
            raise ValueError('direct products require a GUGA CSFSpinBackend')
        if self.backend._ham is not ham or self.backend.nbasis != wfn.nbasis:
            raise ValueError('backend, Hamiltonian and orbital counts must match')
        if strategy not in self._strategies:
            raise ValueError('strategy must be one of '+', '.join(self._strategies))
        if basis_storage not in ('labels', 'packed') or spin_storage not in ('full', 'checkpoints'):
            raise ValueError("basis_storage must be labels/packed and spin_storage full/checkpoints")
        self.basis_storage, self.spin_storage = basis_storage, spin_storage
        self.strategy = strategy
        self.generator_tol = float(generator_tol)
        self.generator_screen = generator_screen
        if not np.isfinite(self.generator_tol) or self.generator_tol < 0:
            raise ValueError('generator_tol must be finite and nonnegative')
        if generator_screen not in ('budget', 'term'):
            raise ValueError("generator_screen must be 'budget' or 'term'")
        if self.generator_tol and strategy not in self._prepared_strategies:
            raise ValueError('generator screening requires a prepared strategy')
        self.product_tol, self.matrix_cutoff = float(product_tol), float(matrix_cutoff)
        if any(not np.isfinite(v) or v < 0 for v in (self.product_tol, self.matrix_cutoff)):
            raise ValueError('direct tolerances must be finite and nonnegative')
        self.nthreads = _integer(self.backend.nthreads if nthreads is None else nthreads, 'nthreads')
        if not 1 <= self.nthreads <= 256:
            raise ValueError('nthreads must be between 1 and 256')
        self.max_product_scratch_bytes = _integer(max_product_scratch_bytes, 'max_product_scratch_bytes')
        if self.max_product_scratch_bytes < 0:
            raise ValueError('max_product_scratch_bytes must be nonnegative')
        if product_accumulation not in ('private', 'gather'):
            raise ValueError('product_accumulation must be private or gather')
        if product_accumulation == 'gather' and strategy not in self._strategies[:3]:
            raise ValueError('gather accumulation currently requires a compact strategy')
        if spin_encoder not in ('auto', 'portable', 'bmi2'):
            raise ValueError('spin_encoder must be auto, portable or bmi2')
        self.product_accumulation, self.spin_encoder = product_accumulation, spin_encoder
        if compact_spin_kernel not in ('general', 'short'):
            raise ValueError('compact_spin_kernel must be general or short')
        self.compact_spin_kernel = compact_spin_kernel
        self.worker_cpus = () if worker_cpus is None else tuple(_integer(c, 'worker CPU') for c in worker_cpus)
        if self.worker_cpus:
            if strategy != 'compact-adaptive-guga':
                raise ValueError('worker_cpus requires compact-adaptive-guga')
            if len(self.worker_cpus) != self.nthreads or len(set(self.worker_cpus)) != self.nthreads:
                raise ValueError('worker_cpus must contain one distinct CPU per requested worker')
            if not hasattr(os, 'sched_getaffinity') or not set(self.worker_cpus) <= os.sched_getaffinity(0):
                raise ValueError('worker_cpus must belong to the allowed Linux CPU set')
        self.max_graph_bytes = _integer(max_graph_bytes, 'max_graph_bytes')
        self.max_path_bytes = _integer(max_path_bytes, 'max_path_bytes')
        self.max_ordered_bytes = _integer(max_ordered_bytes, 'max_ordered_bytes')
        self.max_hamiltonian_bytes = _integer(max_hamiltonian_bytes, 'max_hamiltonian_bytes')
        self.max_symbolic_bytes = _integer(max_symbolic_bytes, 'max_symbolic_bytes')
        self.max_trie_bytes = _integer(max_trie_bytes, 'max_trie_bytes')
        self.profile_couplings = bool(profile_couplings)
        self.collect_spin_stats = bool(collect_spin_stats)
        if not self.collect_spin_stats and strategy == 'semi-direct':
            raise ValueError('semi-direct requires spin statistics')
        self.max_batch_elements = _integer(max_batch_elements, 'max_batch_elements')
        self.batch_min_pairs = _integer(batch_min_pairs, 'batch_min_pairs')
        self.batch_min_terms = _integer(batch_min_terms, 'batch_min_terms')
        if min(self.max_graph_bytes, self.max_path_bytes, self.max_ordered_bytes,
               self.max_hamiltonian_bytes, self.max_symbolic_bytes, self.max_trie_bytes,
               self.max_batch_elements, self.batch_min_pairs, self.batch_min_terms, self.collect_spin_stats) < 0:
            raise ValueError('cache byte budgets must be nonnegative')
        self._ham, self.ecore = ham, self.backend.ecore
        self._sector = (wfn.nbasis, wfn.nelec, wfn.two_s)
        self._labels = tuple(wfn)
        self._settings = self._config()
        setup_started = perf_counter_ns()
        self._native = self._construct(wfn)
        self.setup_seconds = (perf_counter_ns() - setup_started) / 1e9
        self.last_update_stats = {}
        self.last_error_inf = self.last_error_l2 = np.empty(0)
        self.last_product_stats = {}

    def _config(self):
        return (self.strategy, self.product_tol, self.matrix_cutoff, self.nthreads,
                self.max_graph_bytes, self.max_path_bytes, self.max_ordered_bytes,
                self.max_hamiltonian_bytes, self.profile_couplings,
                self.max_batch_elements, self.batch_min_pairs, self.batch_min_terms, self.collect_spin_stats,
                self.generator_tol, self.generator_screen, self.basis_storage, self.spin_storage, self.max_symbolic_bytes,
                self.max_trie_bytes, self.max_product_scratch_bytes, self.product_accumulation,
                self.spin_encoder, self.worker_cpus, self.compact_spin_kernel)

    def _construct(self, wfn):
        from . import _csf
        b = self.backend
        return _csf._DirectHamiltonian(b.one, b.two, wfn._native_basis(self.basis_storage),
            self.nthreads, self._strategies.index(self.strategy), self.product_tol,
            self.matrix_cutoff, ('reference', 'trie', 'residue', 'zobrist').index(b.matrix_index),
            b.max_index_bytes, self.max_graph_bytes, self.max_path_bytes,
            b.max_tensor_entries, b.max_connection_terms, b.max_bond_elements,
            b._guga_mode, b.coupling_reuse, (b.max_reuse_bytes if b.max_reuse_bytes_per_worker is None
                else self.nthreads*b.max_reuse_bytes_per_worker), b.reuse_strategy == 'packed',
            self.max_ordered_bytes, self.max_hamiltonian_bytes, self.profile_couplings,
            self.max_batch_elements, self.batch_min_pairs, self.batch_min_terms, self.collect_spin_stats,
            self.spin_storage == 'checkpoints', self.max_symbolic_bytes, self.max_trie_bytes,
            self.max_product_scratch_bytes, self.product_accumulation == 'gather',
            ('auto', 'portable', 'bmi2').index(self.spin_encoder), self.worker_cpus,
            self.compact_spin_kernel == 'short')

    @property
    def shape(self):
        return (len(self._labels), len(self._labels))

    dtype = property(lambda self: np.dtype(float))
    basis_context = property(lambda self: self._native.basis.context)
    product_info = property(lambda self: dict(strategy=self.strategy, basis_storage=self.basis_storage,
        max_symbolic_bytes=self.max_symbolic_bytes,
        max_trie_bytes=self.max_trie_bytes,
        product_accumulation=self.product_accumulation, spin_encoder=self.spin_encoder,
        worker_cpus=self.worker_cpus,
        compact_spin_kernel=self.compact_spin_kernel,
        spin_storage=self.spin_storage, product_tol=self.product_tol,
        matrix_cutoff=self.matrix_cutoff, nthreads=self.nthreads,
        generator_tol=self.generator_tol,generator_screen=self.generator_screen, **self._native.info))

    def diagonal(self):
        return self._native.diagonal()

    def product_memory_forecast(self, columns=1, *, generator_tol=None, unscreened=False):
        """Worst-case private product storage, before considering input activity.

        The cap covers output/omission replicas and error reduction arrays.
        Returned input/output/metadata estimates are additional storage; memo,
        coupling scratch, allocator overhead and solver storage are separate.
        """
        if self._config() != self._settings:
            raise ValueError('rebuild the direct operator to change product settings')
        columns = _integer(columns, 'columns')
        if columns < 0:
            raise ValueError('columns must be nonnegative')
        tolerance = self.generator_tol if generator_tol is None else float(generator_tol)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('generator_tol must be finite and nonnegative')
        if (tolerance or unscreened) and self.strategy not in self._prepared_strategies:
            raise ValueError('generator screening and unscreened overrides require a prepared strategy')
        exact = unscreened or (self.product_tol == self.matrix_cutoff == tolerance == 0.)
        info = self._native.info
        n, ng = self.shape[0], info['configurations']
        workers = min(self.nthreads, max(1, ng))
        ld = np.dtype(np.longdouble).itemsize
        per_value = 8 + (0 if exact else ld)
        def scratch(owners, width):
            return n * width * owners * per_value + (0 if exact else (3 * owners + 1) * width * ld)
        one = scratch(1, 1)
        gather = self.product_accumulation == 'gather'
        tile_width = columns if gather or not one else min(columns, self.max_product_scratch_bytes // one)
        active_workers = workers
        while not gather and active_workers > 1 and scratch(active_workers, tile_width) > self.max_product_scratch_bytes:
            active_workers -= 1
        return dict(columns=columns, exact=exact, requested_workers=workers,
            workers=active_workers, rhs_tile_width=tile_width,
            fits=gather and exact or (not gather and (not columns or tile_width > 0)),
            uncapped_product_scratch_bytes=0 if gather else scratch(workers, columns),
            forecast_product_scratch_bytes=0 if gather else scratch(active_workers, tile_width),
            max_product_scratch_bytes=self.max_product_scratch_bytes,
            input_bytes=8*n*columns, output_bytes=8*n*columns,
            rhs_tiling_transient_bytes=16*n*tile_width if tile_width<columns else 0,
            coefficient_metadata_bytes_estimate=8*(n+ng)+(0 if exact else ld*ng*tile_width))

    def matmat(self, x, *, generator_tol=None, unscreened=False):
        started = perf_counter_ns()
        if self._config() != self._settings:
            raise ValueError('rebuild the direct operator to change product settings')
        if np.iscomplexobj(x):
            raise TypeError('direct CSF products support real coefficients only')
        x = np.asarray(x, dtype=float, order='C')
        if x.ndim != 2 or x.shape[0] != self.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError('trial block must be finite with shape (basis, columns)')
        tolerance = self.generator_tol if generator_tol is None else float(generator_tol)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('generator_tol must be finite and nonnegative')
        if (tolerance or unscreened) and self.strategy not in self._prepared_strategies:
            raise ValueError('generator screening and unscreened overrides require a prepared strategy')
        validated = perf_counter_ns()
        # Native workers fit the cap at this width. Tile only when even one
        # private output owner would exceed it. Copy each tile contiguously.
        exact = unscreened or (self.product_tol == self.matrix_cutoff == tolerance == 0.)
        n, columns = x.shape
        ld = np.dtype(np.longdouble).itemsize
        per_column = n * (8 if exact else 8 + ld) + (0 if exact else 4 * ld)
        width = columns
        if self.product_accumulation == 'private' and per_column and columns:
            width = min(columns, self.max_product_scratch_bytes // per_column)
            if not width:
                raise MemoryError('max_product_scratch_bytes cannot fit one private output column; '
                                  'increase the cap or use exact compact gather accumulation')
        if columns <= width:
            y, inf, l2, stats = self._native.apply(x, tolerance, ('budget', 'term').index(self.generator_screen), bool(unscreened))
            stats['rhs_tiles'] = 1
        else:
            y, inf, l2 = np.empty_like(x), np.empty(columns), np.empty(columns)
            tile_stats = []
            for begin in range(0, columns, width):
                end = min(columns, begin + width)
                tile = np.ascontiguousarray(x[:, begin:end])
                y[:, begin:end], inf[begin:end], l2[begin:end], current = self._native.apply(
                    tile, tolerance, ('budget', 'term').index(self.generator_screen), bool(unscreened))
                tile_stats.append(dict(current))
            # Preserve the complete records: sums of timings/counters describe
            # work across tiles, maxima describe live footprints and workers.
            peak_keys = {'workers', 'requested_workers', 'team_workers', 'exact_product_kernel',
                         'gather_accumulation', 'optimized', 'source_traversal', 'compact_traversal',
                         'block_traversal', 'late_screening', 'prepared_guga', 'sparse_source_policy',
                         'active_source_groups', 'scratch_workers_limited', 'worker_affinity'}
            stats = {key: (max if key in peak_keys or 'bytes' in key or key.endswith(
                ('_peak_elements', '_cpu_first', '_cpu_last', '_sampling_denominator', '_min', '_median', '_max')) else sum)(
                s.get(key, 0) for s in tile_stats) for key in set().union(*tile_stats)}
            stats.update(rhs_tiles=len(tile_stats), tile_product_stats=tile_stats)
        stats['python_validation_ns'] = validated - started
        stats['native_calls_and_tiling_ns'] = perf_counter_ns() - validated
        stats['python_product_ns'] = perf_counter_ns() - started
        self.last_error_inf, self.last_error_l2 = inf, l2
        self.last_product_stats = dict(stats, generator_tol=0. if unscreened else tolerance,
                                       generator_screen=self.generator_screen, unscreened=bool(unscreened))
        return y

    def matvec(self, x, out=None, *, generator_tol=None, unscreened=False):
        started = perf_counter_ns()
        x = _real_vector(x, self.shape[1])
        validated = perf_counter_ns()
        y = self.matmat(x[:, None], generator_tol=generator_tol, unscreened=unscreened)[:, 0]
        if out is not None:
            if not isinstance(out, np.ndarray) or out.shape != y.shape or out.dtype != np.float64:
                raise ValueError('out must be a float64 array of the output shape')
            out[:] = y
            y = out
        self.last_product_stats['python_matvec_validation_ns'] = validated - started
        self.last_product_stats['python_matvec_ns'] = perf_counter_ns() - started
        return y

    __call__ = matvec

    def update(self, ham, wfn):
        """Rebuild occupation/path metadata after append, publishing on success.

        Direct modes rebuild metadata; semi-direct also rebuilds its bounded
        value cache. Changed prefixes/sectors and Hamiltonian objects require
        explicit reconstruction, as for csf_op.
        """
        started = perf_counter_ns()
        if self._config() != self._settings:
            raise ValueError('rebuild the direct operator to change product settings')
        if (not isinstance(wfn, csf_wfn) or ham is not self._ham or
                (wfn.nbasis, wfn.nelec, wfn.two_s) != self._sector or
                tuple(wfn)[:len(self._labels)] != self._labels):
            raise ValueError('update requires the original Hamiltonian and unchanged CSF prefix')
        if len(wfn) == len(self._labels):
            self.last_update_stats = dict(rebuilt=False, seconds=(perf_counter_ns() - started) / 1e9)
            return
        previous = self.product_info['persistent_bytes_estimate']
        native = self._construct(wfn)
        overlap = previous + native.info['persistent_bytes_estimate']
        self._native, self._labels = native, tuple(wfn)
        self.last_update_stats = dict(rebuilt=True, seconds=(perf_counter_ns() - started) / 1e9,
            old_new_persistent_bytes_estimate=overlap)
        self.last_product_stats = {}
        self.last_error_inf = self.last_error_l2 = np.empty(0)
