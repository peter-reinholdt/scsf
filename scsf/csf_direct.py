"""Direct and bounded semi-direct GUGA Hamiltonian products.

The stored native CSR remains the production default. Contribution screening
is nonlinear; disable all screening for exact linear products and residual checks.
"""
import numpy as np

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
                 max_symbolic_bytes=16*1024*1024, max_trie_bytes=16*1024*1024):
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
        self._native = self._construct(wfn)
        self.last_error_inf = self.last_error_l2 = np.empty(0)
        self.last_product_stats = {}

    def _config(self):
        return (self.strategy, self.product_tol, self.matrix_cutoff, self.nthreads,
                self.max_graph_bytes, self.max_path_bytes, self.max_ordered_bytes,
                self.max_hamiltonian_bytes, self.profile_couplings,
                self.max_batch_elements, self.batch_min_pairs, self.batch_min_terms, self.collect_spin_stats,
                self.generator_tol, self.generator_screen, self.basis_storage, self.spin_storage, self.max_symbolic_bytes,
                self.max_trie_bytes)

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
            self.spin_storage == 'checkpoints', self.max_symbolic_bytes, self.max_trie_bytes)

    @property
    def shape(self):
        return (len(self._labels), len(self._labels))

    dtype = property(lambda self: np.dtype(float))
    basis_context = property(lambda self: self._native.basis.context)
    product_info = property(lambda self: dict(strategy=self.strategy, basis_storage=self.basis_storage,
        max_symbolic_bytes=self.max_symbolic_bytes,
        max_trie_bytes=self.max_trie_bytes,
        spin_storage=self.spin_storage, product_tol=self.product_tol,
        matrix_cutoff=self.matrix_cutoff, nthreads=self.nthreads,
        generator_tol=self.generator_tol,generator_screen=self.generator_screen, **self._native.info))

    def diagonal(self):
        return self._native.diagonal()

    def matmat(self, x, *, generator_tol=None, unscreened=False):
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
        y, inf, l2, stats = self._native.apply(x, tolerance, ('budget', 'term').index(self.generator_screen), bool(unscreened))
        self.last_error_inf, self.last_error_l2 = inf, l2
        self.last_product_stats = dict(stats, generator_tol=0. if unscreened else tolerance,
                                       generator_screen=self.generator_screen, unscreened=bool(unscreened))
        return y

    def matvec(self, x, out=None, *, generator_tol=None, unscreened=False):
        x = _real_vector(x, self.shape[1])
        y = self.matmat(x[:, None], generator_tol=generator_tol, unscreened=unscreened)[:, 0]
        if out is not None:
            if not isinstance(out, np.ndarray) or out.shape != y.shape or out.dtype != np.float64:
                raise ValueError('out must be a float64 array of the output shape')
            out[:] = y
            return out
        return y

    __call__ = matvec

    def update(self, ham, wfn):
        """Rebuild occupation/path metadata after append, publishing on success.

        Direct modes rebuild metadata; semi-direct also rebuilds its bounded
        value cache. Changed prefixes/sectors and Hamiltonian objects require
        explicit reconstruction, as for csf_op.
        """
        if self._config() != self._settings:
            raise ValueError('rebuild the direct operator to change product settings')
        if (not isinstance(wfn, csf_wfn) or ham is not self._ham or
                (wfn.nbasis, wfn.nelec, wfn.two_s) != self._sector or
                tuple(wfn)[:len(self._labels)] != self._labels):
            raise ValueError('update requires the original Hamiltonian and unchanged CSF prefix')
        if len(wfn) == len(self._labels):
            return
        native = self._construct(wfn)
        self._native, self._labels = native, tuple(wfn)
        self.last_product_stats = {}
        self.last_error_inf = self.last_error_l2 = np.empty(0)
