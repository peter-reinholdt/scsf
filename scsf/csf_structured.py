"""Exact structured CSF products with explicit operator-link factorizations."""
import os
from threading import RLock
from time import perf_counter
import numpy as np
from .csf import csf_wfn, _integer, _limit
from .config import get_num_threads
from .integrals import Hamiltonian, _StructuredIntegrals


class csf_structured_op:
    """Exact electronic Hamiltonian products in an arbitrary selected CSF space.

    The available kernels are N−2, CIH, N−1 and particle–hole. The
    operator stores selected occupation blocks and applies the electronic
    Hamiltonian without assembling its off-diagonal matrix. Dense, packed,
    Cholesky and THC integrals use the same interface.

    The shared memory budget covers native kernel metadata, coefficients,
    caches, HCI transfer tables and product arrays. N−2 supports private
    outputs and owner gather.
    """
    kernels = ('nminus2', 'cih', 'nminus1', 'particle-hole')

    def __init__(self, ham, wfn, *, strategy='nminus2', nthreads=None, max_mem=None,
                 worker_cpus=None, pair_accumulation=None, cih_slot_sweep=False,
                 incremental_removals=True):
        if not isinstance(wfn, csf_wfn):
            raise TypeError('csf_structured_op requires csf_wfn')
        if strategy not in self.kernels:
            raise ValueError('unknown Hamiltonian kernel')
        if cih_slot_sweep and strategy != 'cih':
            raise ValueError('cih_slot_sweep requires the CIH kernel')
        self.cih_slot_sweep = bool(cih_slot_sweep)
        self.incremental_removals = bool(incremental_removals)
        if pair_accumulation is None:
            pair_accumulation = 'private' if strategy == 'nminus2' else 'gather'
        if pair_accumulation not in ('gather', 'private') or (pair_accumulation == 'private' and strategy != 'nminus2'):
            raise ValueError("pair_accumulation must be 'gather', or 'private' for nminus2")
        compact = isinstance(ham, _StructuredIntegrals)
        if not compact and not isinstance(ham, Hamiltonian):
            raise TypeError('expected dense or factored Hamiltonian integrals')
        if ham.nbasis != wfn.nbasis:
            raise ValueError('Hamiltonian and CSF orbital counts must match')
        self.nthreads = get_num_threads() if nthreads is None else _integer(nthreads, 'nthreads')
        if not 1 <= self.nthreads <= 256:
            raise ValueError('nthreads must be between 1 and 256')
        self.worker_cpus = tuple(_integer(v, 'worker CPU') for v in (() if worker_cpus is None else worker_cpus))
        if self.worker_cpus and (len(self.worker_cpus) != self.nthreads or
                len(set(self.worker_cpus)) != self.nthreads or not hasattr(os, 'sched_getaffinity') or
                not set(self.worker_cpus) <= os.sched_getaffinity(0)):
            raise ValueError('worker_cpus must contain one distinct allowed CPU per worker')
        self.max_mem = _limit(max_mem, 'max_mem')
        if self.max_mem < 0:
            raise ValueError('max_mem must be nonnegative')
        self.strategy, self._ham = strategy, ham
        self.pair_accumulation = pair_accumulation
        self.ecore = ham.ecore
        self._integrals = ((ham.one_mo, *ham._structured_eri()) if compact
                           else (ham.one_mo, ham.two_mo, None))
        from . import _csf
        self._transfer_bounds = None if compact else _csf._HCITransferBounds(
            ham.one_mo, ham.two_mo)
        self._growth_lock = RLock()
        self._rdm_cache = None
        self._sector = (wfn.nbasis, wfn.nelec, wfn.two_s)
        self._settings = self._config()
        start = perf_counter()
        self._state = self._construct(wfn._occupation_basis())
        self.setup_seconds = perf_counter()-start
        self.last_product_stats = {}

    def _config(self):
        return (self.strategy, self.nthreads, self.max_mem, self.worker_cpus,
                self.pair_accumulation, self.cih_slot_sweep, self.incremental_removals)

    def _construct(self, basis, previous=None):
        from . import _csf
        try:
            native = _csf._StructuredHamiltonian(*self._integrals, basis, self.kernels.index(self.strategy), self.nthreads,
                self.max_mem, self.worker_cpus, self.pair_accumulation == 'private', self._transfer_bounds,
                previous, self.incremental_removals, self.cih_slot_sweep)
        except _csf.BudgetError as exc:
            raise _csf.BudgetError(f'{self.strategy} setup with max-mem={self.max_mem} bytes: {exc}') from None
        return basis, native, basis.public_rows

    @property
    def shape(self):
        size = len(self._state[0])
        return size, size

    dtype = np.dtype(float)

    @property
    def basis_context(self):
        return dict(nbasis=self._sector[0], nelec=self._sector[1], two_s=self._sector[2],
                    phase_convention='cg-real-alpha-beta-v1', generation=self._state[0].info['generation'])

    @property
    def product_info(self):
        return dict(strategy=self.strategy, nthreads=self.nthreads, worker_cpus=self.worker_cpus,
                    pair_accumulation=self.pair_accumulation, **self._state[1].info)

    def labels(self, *, native=False):
        """Iterate basis labels on demand in public or native row order."""
        basis = self._state[0]
        lookup = basis.native_label if native else basis.public_label
        return (lookup(row) for row in range(len(basis)))

    def native_labels(self, rows):
        """Get labels for specific native rows without enumerating the space."""
        lookup = self._state[0].native_label
        return tuple(lookup(int(row)) for row in rows)

    def with_basis(self, wfn, *, concurrent=False):
        """Construct an independent operator on related CSFs."""
        max_mem = self.max_mem
        if concurrent:
            self._state[1].release_product_workspace()
            max_mem -= self.product_info['tracked_memory_bytes']
        return type(self)(self._ham, wfn, strategy=self.strategy, nthreads=self.nthreads,
            worker_cpus=self.worker_cpus, max_mem=max_mem, pair_accumulation=self.pair_accumulation,
            cih_slot_sweep=self.cih_slot_sweep, incremental_removals=self.incremental_removals)

    def to_native_order(self, values):
        """Convert a public-order vector or column block to native order."""
        basis, _, rows = self._state
        values = np.asarray(values)
        if values.ndim not in (1, 2) or len(values) != len(basis):
            raise ValueError('expected one public row per CSF')
        if not rows.size:
            return values
        reordered = np.empty_like(values)
        reordered[rows] = values
        return reordered

    def diagonal(self):
        _, native, rows = self._state
        diagonal = native.diagonal()
        return diagonal[rows] if rows.size else diagonal

    def native_diagonal(self):
        return self._state[1].diagonal()

    def _matmat(self, x, *, canonical):
        if self._settings != self._config():
            raise ValueError('rebuild the operator to change structured settings')
        if np.iscomplexobj(x):
            raise TypeError('structured products require real coefficients')
        x = np.asarray(x, dtype=float, order='C')
        basis, native, rows = self._state
        size = len(basis)
        if x.ndim != 2 or x.shape[0] != size or not np.all(np.isfinite(x)):
            raise ValueError('trial block must be finite with shape (basis, columns)')
        permuted = not canonical and rows.size
        if permuted:
            reordered = np.empty_like(x)
            reordered[rows] = x
            x = reordered
        result, stats = native.apply(x)
        self.last_product_stats = dict(stats)
        return result[rows] if permuted else result

    def matmat(self, x):
        return self._matmat(x, canonical=False)

    def native_matmat(self, x):
        return self._matmat(x, canonical=True)

    def select_hci(self, weights, eps, *, selection_unit='csf', stop_growth_fraction=0.):
        """Grow with strict HCI using absolute native-row weights.

        Return the old-to-new row map and statistics. An empty map represents
        an unchanged prefix. Publish after the new operator is ready.
        Skip growth when the proposed CSF count is at most this fraction of
        the current dimension. The statistics distinguish proposed and added CSFs.
        """
        if np.iscomplexobj(weights):
            raise TypeError('HCI weights must be real')
        weights = np.asarray(weights, dtype=float, order='C')
        if not 0. <= stop_growth_fraction <= 1.:
            raise ValueError('stop_growth_fraction must be in [0, 1]')
        with self._growth_lock:
            if self._settings != self._config():
                raise ValueError('rebuild the operator to change structured settings')
            if selection_unit not in ('csf', 'configuration'):
                raise ValueError('selection_unit must be csf or configuration')
            old = self._state[1]
            complete = selection_unit == 'configuration'
            if complete:
                info = old.info
                if info['configurations'] != info['complete_configurations']:
                    raise ValueError('configuration selection requires a complete starting space')
            basis, remap, stats = old.select_hci(weights, eps, complete, stop_growth_fraction)
            if basis is not None:
                start = perf_counter()
                replacement = self._construct(basis, previous=old)
                constructed = perf_counter()
                replacement[1].reuse_diagonal(old, remap)
                ready = perf_counter()
                stats['replacement_ns'] = round((constructed-start)*1e9)
                stats['diagonal_reuse_ns'] = round((ready-constructed)*1e9)
                self._state = replacement
                self._rdm_cache = None
                self.setup_seconds = perf_counter()-start
                self.last_product_stats = {}
        return remap, dict(stats)

    def native_rdms(self, coefficients, *, max_output_bytes=256*1024**2,
                    max_link_bytes=None, max_workspace_bytes=None, cache=False):
        """Return spin-free 1- and 2-RDMs of a real native-row vector and memory counters.

        The 2-RDM uses the same physicist order as the Hamiltonian, so its
        two-electron energy contribution is `0.5 * np.sum(two * two_rdm)`.
        Both density matrices scale with the squared norm of the input.
        With `cache=True` the last request for the current space is retained.
        Returned arrays can be modified independently of the cache.
        """
        if np.iscomplexobj(coefficients):
            raise TypeError('density matrix coefficients must be real')
        x = np.asarray(coefficients, dtype=float, order='C')
        budgets = (max_output_bytes,
            self.max_mem if max_link_bytes is None else max_link_bytes,
            self.max_mem if max_workspace_bytes is None else max_workspace_bytes,
            self.max_mem)
        budgets = tuple(_integer(value, 'density matrix budget') for value in budgets)
        if min(budgets) < 0:
            raise ValueError('density matrix budgets must be nonnegative')
        if not cache:
            return self._state[1].rdms(x, *budgets)
        with self._growth_lock:
            native = self._state[1]
            entry = self._rdm_cache
            if entry is not None:
                prior, limits, saved, one, two, stats = entry
                if prior is native and limits == budgets and np.array_equal(x, saved):
                    return one.copy(), two.copy(), dict(stats, cache_hit=1)
            one, two, stats = native.rdms(x, *budgets)
            self._rdm_cache = native, budgets, x.copy(), one.copy(), two.copy(), stats.copy()
            return one, two, dict(stats, cache_hit=0)

    def matvec(self, x):
        if np.iscomplexobj(x):
            raise TypeError('structured products require real coefficients')
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != self.shape[0]:
            raise ValueError('trial vector must have shape (basis,)')
        return self.matmat(x[:, None])[:, 0]

    __call__ = matvec

    def update(self, ham, wfn):
        if not isinstance(wfn, csf_wfn):
            raise TypeError('csf_structured_op requires csf_wfn')
        if self._settings != self._config():
            raise ValueError('rebuild the operator to change structured settings')
        if ham is not self._ham or (wfn.nbasis, wfn.nelec, wfn.two_s) != self._sector:
            raise ValueError('updated operator must retain Hamiltonian and CSF sector')
        with self._growth_lock:
            size = self.shape[0]
            if len(wfn) < size or any(label != wfn[i] for i, label in enumerate(self.labels())):
                raise ValueError('updates must append to the original CSF order')
            if len(wfn) == size:
                return
            start = perf_counter()
            self._state = self._construct(wfn._occupation_basis(), previous=self._state[1])
            self._rdm_cache = None
            self.setup_seconds = perf_counter()-start
            self.last_product_stats = {}

    def append_selected_labels(self, wfn, old_size):
        """Bring the public label container up to date after native selection."""
        basis = self._state[0]
        if len(wfn) != old_size or old_size > len(basis) or (wfn.nbasis, wfn.nelec, wfn.two_s) != self._sector:
            raise ValueError('label container does not match the previous occupation space')
        wfn._append_native([basis.public_label(row) for row in range(old_size, len(basis))])
        wfn._occupation_snapshot = basis
