# SCSF; licensed under GPL-3.0-or-later.
"""Spin-adapted CI with native spin contractions and a determinant oracle.

Orbitals are coupled in their supplied order. Steps ``0,u,d,2`` denote empty, spin-up-coupled,
spin-down-coupled, and doubly occupied orbitals; ``u/d`` are not alpha/beta spins.
All determinant coefficients use alpha-before-beta creation order and the real
Clebsch--Gordan convention documented in docs/conventions.md.
"""
from collections import OrderedDict
from itertools import combinations, combinations_with_replacement
import json
from math import comb, sqrt
from operator import index
import numpy as np
from scipy import sparse
from .expansion import DeterminantExpansion
_popcount = int.bit_count

def _integer(value, name):
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc

def _limit(value, name):
    """None disables a work limit up to the native platform's index capacity."""
    return int(np.iinfo(np.uintp).max) if value is None else _integer(value, name)

def _choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0

def _sector(nbasis, nelec, two_s):
    nbasis, nelec, two_s = (_integer(v, n) for v, n in zip((nbasis, nelec, two_s), ('nbasis', 'nelec', 'two_s')))
    if nbasis < 0 or not 0 <= nelec <= 2 * nbasis:
        raise ValueError('require nbasis >= 0 and 0 <= nelec <= 2*nbasis')
    if two_s < 0 or two_s > min(nelec, 2 * nbasis - nelec) or (nelec - two_s) % 2:
        raise ValueError('two_s is incompatible with the electron/orbital counts')
    return (nbasis, nelec, two_s)

def csf_dimension(nbasis, nelec, two_s):
    """Return the exact fixed-spin dimension, using Python's unbounded integers."""
    nbasis, nelec, two_s = _sector(nbasis, nelec, two_s)
    a, b = ((nelec + two_s) // 2, (nelec - two_s) // 2)
    return _choose(nbasis, a) * _choose(nbasis, b) - _choose(nbasis, a + 1) * _choose(nbasis, b - 1)

def _configuration(label):
    return tuple((0 if s == '0' else 2 if s == '2' else 1 for s in label))

def _paths(occupations, two_s):
    """Generate paths in canonical 0,u,d,2 order, pruning unreachable spins."""
    remaining = [0] * (len(occupations) + 1)
    for p in range(len(occupations) - 1, -1, -1):
        remaining[p] = remaining[p + 1] + (occupations[p] == 1)

    def visit(p, spin, label):
        if abs(spin - two_s) > remaining[p] or (spin + remaining[p] - two_s) % 2:
            return
        if p == len(occupations):
            yield label
        elif occupations[p] != 1:
            yield from visit(p + 1, spin, label + str(occupations[p]))
        else:
            yield from visit(p + 1, spin + 1, label + 'u')
            if spin:
                yield from visit(p + 1, spin - 1, label + 'd')
    yield from visit(0, 0, '')

class csf_wfn:
    """Append-only genealogical CSFs at fixed electron number and total spin.

    Labels are strings or sequences of the symbols ``0,u,d,2``. Exact labels,
    rather than hashes alone, determine identity. Metadata and orbital order
    are immutable. Basis files contain the sector and ordered labels.
    """

    def __init__(self, nbasis, nelec, two_s):
        self._nbasis, self._nelec, self._two_s = _sector(nbasis, nelec, two_s)
        self._labels = []
        self._indices = {}
        self._configs = {}
        self._native_basis_snapshot = None
    nbasis = property(lambda self: self._nbasis)
    nelec = property(lambda self: self._nelec)
    two_s = property(lambda self: self._two_s)
    ncsf = property(lambda self: len(self))
    nconfig = property(lambda self: len(self._configs))

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return iter(self._labels)

    def __getitem__(self, item):
        return self._labels[item] if not isinstance(item, slice) else tuple(self._labels[item])

    def __copy__(self):
        duplicate = type(self)(self.nbasis, self.nelec, self.two_s)
        for label in self:
            duplicate.add_csf(label)
        return duplicate

    def __deepcopy__(self, memo):
        duplicate = self.__copy__()
        memo[id(self)] = duplicate
        return duplicate

    def _validate(self, steps):
        label = ''.join(steps)
        if len(label) != self.nbasis or any((s not in '0ud2' for s in label)):
            raise ValueError('a CSF needs exactly nbasis symbols from 0,u,d,2')
        spin = 0
        for step in label:
            spin += (step == 'u') - (step == 'd')
            if spin < 0:
                raise ValueError('negative intermediate spin in CSF path')
        if spin != self.two_s or sum(_configuration(label)) != self.nelec:
            raise ValueError('CSF has the wrong electron number or final spin')
        return label

    def add_csf(self, steps):
        """Append one validated label; return its index (also for duplicates)."""
        label = self._validate(steps)
        if label not in self._indices:
            self._indices[label] = len(self)
            self._configs.setdefault(_configuration(label), []).append(len(self))
            self._labels.append(label)
            self._native_basis_snapshot = None
        return self._indices[label]

    def _native_basis(self, storage='labels'):
        """Share an immutable native snapshot across direct coupling kernels.

        Generation is the append count; snapshot identity also distinguishes
        unrelated bases with equal counts. Existing operators keep their old
        snapshot after append. Native storage preserves the public vector order.
        Only the most recently requested storage format is cached here.
        """
        if storage not in ('labels', 'packed'):
            raise ValueError("basis storage must be 'labels' or 'packed'")
        snapshot = self._native_basis_snapshot
        if snapshot is None or bool(snapshot.storage_info['packed_spin_paths']) != (storage == 'packed'):
            from . import _csf
            snapshot = _csf._SelectedBasis(self.nbasis, self.nelec, self.two_s, tuple(self), len(self), storage == 'packed')
            self._native_basis_snapshot = snapshot
        return snapshot

    def _append_native(self, labels, groups):
        """Publish a validated native selection, preparing all containers first.

        Private trusted route: native search has checked the sector, uniqueness,
        immutable-parent snapshot, and guards. Group indices are final append
        indices. Public insertion and the Python oracle retain full validation.
        """
        if not labels:
            return
        indices = self._indices.copy()
        indices.update(zip(labels, range(len(self), len(self) + len(labels))))
        configurations = self._configs.copy()
        for occupation, positions in groups:
            configurations[occupation] = self._configs.get(occupation, []) + positions
        appended = self._labels + labels
        self._labels, self._indices, self._configs = (appended, indices, configurations)
        self._native_basis_snapshot = None

    def index_csf(self, steps):
        """Return the label's index, or -1 if absent."""
        return self._indices.get(self._validate(steps), -1)

    def configuration_indices(self, occupations):
        """Return immutable global indices, including noncontiguous appends."""
        return tuple(self._configs.get(self._occupations(occupations), ()))

    def _occupations(self, occupations):
        occ = tuple((_integer(v, 'occupation') for v in occupations))
        if len(occ) != self.nbasis or any((v not in (0, 1, 2) for v in occ)) or sum(occ) != self.nelec:
            raise ValueError('occupations must have length nbasis, values 0..2 and sum nelec')
        return occ

    def add_configuration(self, occupations, *, max_csfs=100000):
        """Append all allowed couplings of an occupation; return number added."""
        occ = self._occupations(occupations)
        m = occ.count(1)
        k = (m - self.two_s) // 2
        count = 0 if m < self.two_s or (m - self.two_s) % 2 else _choose(m, k) - _choose(m, k - 1)
        max_csfs = _limit(max_csfs, 'max_csfs')
        if len(self) + count - len(self._configs.get(occ, ())) > max_csfs:
            raise MemoryError('CSF enumeration exceeds max_csfs')
        old = len(self)
        for label in _paths(occ, self.two_s):
            self.add_csf(label)
        return len(self) - old

    def add_all_csfs(self, *, max_csfs=100000):
        """Enumerate the entire sector, with a dimension check before allocation."""
        max_csfs = _limit(max_csfs, 'max_csfs')
        if csf_dimension(self.nbasis, self.nelec, self.two_s) > max_csfs:
            raise MemoryError('full spin sector exceeds max_csfs')

        def configurations(p, left, occ):
            if p == self.nbasis:
                if left == 0:
                    yield occ
                return
            for n in (0, 1, 2):
                if 0 <= left - n <= 2 * (self.nbasis - p - 1):
                    yield from configurations(p + 1, left - n, occ + (n,))
        old = len(self)
        for occ in configurations(0, self.nelec, ()):
            self.add_configuration(occ, max_csfs=max_csfs)
        return len(self) - old

    def to_step_array(self):
        """Return a copy of the labels as a (ncsf, nbasis) character array."""
        return np.array([list(s) for s in self], dtype='U1').reshape(len(self), self.nbasis)

    def to_configuration_array(self):
        """Return spatial occupations, one row per CSF (duplicates retained)."""
        array = np.array([_configuration(s) for s in self], dtype=np.int8)
        return array.reshape(len(self), self.nbasis)

    def to_file(self, filename):
        """Write the sector and ordered CSF labels as a plain basis file."""
        payload = dict(nbasis=self.nbasis, nelec=self.nelec, two_s=self.two_s, labels=list(self))
        with open(filename, 'w', encoding='utf8') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')

    @classmethod
    def from_file(cls, filename):
        """Read a basis file containing exactly the sector and ordered labels."""
        with open(filename, encoding='utf8') as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or set(data) != {'nbasis', 'nelec', 'two_s', 'labels'}:
            raise ValueError('basis file must contain nbasis, nelec, two_s and labels')
        wfn = cls(data['nbasis'], data['nelec'], data['two_s'])
        for label in data['labels']:
            if wfn.index_csf(label) != -1:
                raise ValueError('duplicate CSF in basis file')
            wfn.add_csf(label)
        return wfn

    def determinant_transform(self, *, two_ms=None, max_determinants=100000, max_expansion_nnz=1000000):
        """Return ``(DeterminantExpansion, U)`` with sparse ``U[det, csf]``.

        Each column contains its complete expansion; no coefficient pruning is
        performed. Guards bound the determinant union and a conservative count
        of expansion entries, including spin assignments that ultimately vanish.
        The default projection is M=S; a different M changes only conversion.
        Export convention requires nonnegative M (nalpha >= nbeta).
        """
        two_ms = self.two_s if two_ms is None else _integer(two_ms, 'two_ms')
        if not 0 <= two_ms <= self.two_s or (two_ms - self.two_s) % 2:
            raise ValueError('two_ms must satisfy 0 <= two_ms <= two_s with matching parity (export requires nalpha >= nbeta)')
        count = sum((_support_bound(s, two_ms) for s in self))
        if count > max_expansion_nnz:
            raise MemoryError('determinant expansion exceeds max_expansion_nnz')
        support = {}
        rows, cols, values = ([], [], [])
        for j, label in enumerate(self):
            for det, value in _expand(label, two_ms).items():
                if det not in support:
                    if len(support) >= max_determinants:
                        raise MemoryError('determinant union exceeds max_determinants')
                    support[det] = len(support)
                rows.append(support[det])
                cols.append(j)
                values.append(value)
        a, b = ((self.nelec + two_ms) // 2, (self.nelec - two_ms) // 2)
        wordbits = np.dtype(np.uint64).itemsize * 8
        nword = (self.nbasis + wordbits - 1) // wordbits
        array = np.zeros((len(support), 2, nword), dtype=np.uint64)
        mask = (1 << wordbits) - 1
        for det, i in support.items():
            for spin, bits in enumerate((det & (1 << self.nbasis) - 1, det >> self.nbasis)):
                for w in range(nword):
                    array[i, spin, w] = bits >> w * wordbits & mask
        det_wfn = DeterminantExpansion(self.nbasis, a, b, array)
        return (det_wfn, sparse.csr_matrix((values, (rows, cols)), shape=(len(support), len(self))))

    def to_determinants(self, coeffs, **kwargs):
        """Export a data-only determinant basis and real coefficients for validation."""
        coeffs = _real_vector(coeffs, len(self))
        det_wfn, transform = self.determinant_transform(**kwargs)
        return (det_wfn, np.asarray(transform @ coeffs))

def _support_bound(label, two_ms):
    m = label.count('u') + label.count('d')
    return _choose(m, (m + two_ms) // 2)

def _expand(label, two_ms):
    """Expand one path; use orbital-major then alpha-before-beta phases."""
    singles = [p for p, s in enumerate(label) if s in 'ud']
    doubles = [p for p, s in enumerate(label) if s == '2']
    nbasis = len(label)
    base = sum((1 << p | 1 << nbasis + p for p in doubles))
    result = {}
    for alpha in combinations(singles, (len(singles) + two_ms) // 2):
        alpha = set(alpha)
        spin, projection, amplitude = (0, 0, 1.0)
        det = base
        for p in singles:
            sigma = 1 if p in alpha else -1
            if label[p] == 'u':
                factor = (spin + sigma * projection + 2) / (2 * (spin + 1))
                sign = 1
                new_spin = spin + 1
            else:
                factor = (spin - sigma * projection) / (2 * (spin + 1))
                sign = -1 if sigma == 1 else 1
                new_spin = spin - 1
            if factor <= 0:
                amplitude = 0.0
                break
            amplitude *= sign * sqrt(factor)
            spin, projection = (new_spin, projection + sigma)
            det |= 1 << (p if sigma == 1 else nbasis + p)
        if amplitude:
            betas = [p for p in range(nbasis) if det & 1 << nbasis + p]
            alphas = [p for p in range(nbasis) if det & 1 << p]
            inversions = sum((q > p for p in betas for q in alphas))
            result[det] = amplitude * (-1 if inversions % 2 else 1)
    return result

def _real_vector(values, size):
    if np.iscomplexobj(values):
        raise TypeError('the reference CSF backend supports real coefficients only')
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f'coefficients must be a finite vector of length {size}')
    return values

def _bits(bits):
    while bits:
        low = bits & -bits
        yield (low.bit_length() - 1)
        bits ^= low

def _act(det, annihilate, create):
    """Apply annihilators/creators in their execution order, returning phase."""
    phase = 1
    for p in annihilate:
        if not det & 1 << p:
            return 0
        phase *= -1 if _popcount(det & (1 << p) - 1) % 2 else 1
        det ^= 1 << p
    for p in create:
        if det & 1 << p:
            return 0
        phase *= -1 if _popcount(det & (1 << p) - 1) % 2 else 1
        det ^= 1 << p
    return phase

class _CSFIntegralBackend:
    """Shared spatial-integral snapshot and CSF block validation."""

    def __init__(self, ham, *, max_block_elements=1000000):
        self._ham = ham
        self.nbasis = ham.nbasis
        self.ecore = float(ham.ecore)
        self.one = np.array(ham.one_mo, copy=True)
        self.two = np.array(ham.two_mo, copy=True)
        k = self.nbasis
        if self.one.shape != (k, k) or self.two.shape != (k,) * 4 or np.iscomplexobj(self.one) or np.iscomplexobj(self.two):
            raise ValueError('reference backend requires common real spatial integrals')
        if not all((np.all(np.isfinite(v)) for v in (self.ecore, self.one, self.two))):
            raise ValueError('integrals must be finite')
        g = self.two.transpose(0, 2, 1, 3)
        if not (np.allclose(self.one, self.one.T, rtol=0, atol=1e-12) and all((np.allclose(g, other, rtol=0, atol=1e-12) for other in (g.swapaxes(0, 1), g.swapaxes(2, 3), g.transpose(2, 3, 0, 1))))):
            raise ValueError('integrals must have real spin-free permutation symmetry')
        self.one.flags.writeable = self.two.flags.writeable = False
        self.max_block_elements = _limit(max_block_elements, 'max_block_elements')
        if self.max_block_elements < 0:
            raise ValueError('max_block_elements must be nonnegative')

    def _validate_block(self, rows, cols, two_s):
        rows, cols = (tuple(rows), tuple(cols))
        if len(rows) * len(cols) > self.max_block_elements:
            raise MemoryError('CSF block exceeds max_block_elements')
        if rows or cols:
            first = rows[0] if rows else cols[0]
            validator = csf_wfn(self.nbasis, sum(_configuration(first)), two_s)
            for label in rows + cols:
                validator._validate(label)
        return (rows, cols)

    def element(self, bra, ket, two_s):
        """Return a signed electronic CSF matrix element."""
        return float(self.block((bra,), (ket,), two_s)[0, 0])

class CSFReferenceBackend(_CSFIntegralBackend):
    """Unpruned Slater--Condon elements transformed to complete CSF supports.

    Integral arrays are snapshotted. Only real, symmetric spatial integrals
    are accepted; ``two_mo[p,r,q,s] = (pq|rs)``. Caches have a total expansion
    entry budget. Dense temporary determinant blocks have an allocation guard.
    """

    def __init__(self, ham, *, max_expansion_nnz=1000000, max_block_elements=1000000, cache_expansion_nnz=100000):
        super().__init__(ham, max_block_elements=max_block_elements)
        self.max_expansion_nnz = _limit(max_expansion_nnz, 'max_expansion_nnz')
        self.cache_expansion_nnz = _integer(cache_expansion_nnz, 'cache_expansion_nnz')
        if self.max_expansion_nnz < 0 or self.cache_expansion_nnz < 0:
            raise ValueError('expansion/cache limits must be nonnegative')
        self._cache = OrderedDict()
        self._cache_size = 0

    def _expansion(self, label, two_s):
        key = (label, two_s)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if _support_bound(label, two_s) > self.max_expansion_nnz:
            raise MemoryError('single-CSF expansion exceeds max_expansion_nnz')
        value = _expand(label, two_s)
        if len(value) <= self.cache_expansion_nnz:
            while self._cache_size + len(value) > self.cache_expansion_nnz and self._cache:
                _, old = self._cache.popitem(last=False)
                self._cache_size -= len(old)
            self._cache[key] = value
            self._cache_size += len(value)
        return value

    def determinant_element(self, bra, ket):
        """Return an electronic element between alpha-before-beta bitstrings."""
        bra, ket = (_integer(bra, 'bra'), _integer(ket, 'ket'))
        if bra < 0 or ket < 0 or (bra | ket) >> 2 * self.nbasis:
            raise ValueError('determinant bits must lie within the spin-orbital basis')
        holes, particles = (list(_bits(ket & ~bra)), list(_bits(bra & ~ket)))
        if len(holes) != len(particles) or len(holes) > 2:
            return 0.0
        k, h, v = (self.nbasis, self.one, self.two)

        def integral(p, r, q, s):
            return v[p % k, r % k, q % k, s % k] if p // k == q // k and r // k == s // k else 0.0
        if not holes:
            occ = list(_bits(ket))
            value = sum((h[p % k, p % k] for p in occ))
            for i, p in enumerate(occ):
                for q in occ[:i]:
                    value += integral(p, q, p, q) - integral(p, q, q, p)
            return value
        if len(holes) == 1:
            q, p = (holes[0], particles[0])
            value = h[p % k, q % k] if p // k == q // k else 0.0
            for r in _bits(ket & bra):
                value += integral(p, r, q, r) - integral(p, r, r, q)
            return _act(ket, (q,), (p,)) * value
        q, s = holes
        p, r = particles
        return _act(ket, (q, s), (r, p)) * (integral(p, r, q, s) - integral(p, r, s, q))

    def block(self, rows, cols, two_s):
        """Return a selected CSF subblock, without filling missing spin paths.

        Labels are validated to belong to one N,S sector and orbital convention.
        """
        rows, cols = self._validate_block(rows, cols, two_s)
        if not rows or not cols:
            return np.zeros((len(rows), len(cols)))
        if sum((_support_bound(s, two_s) for s in rows + cols)) > self.max_expansion_nnz:
            raise MemoryError('block expansions exceed max_expansion_nnz')
        left = [self._expansion(s, two_s) for s in rows]
        right = [self._expansion(s, two_s) for s in cols]
        ld = dict.fromkeys((d for expansion in left for d in expansion))
        rd = dict.fromkeys((d for expansion in right for d in expansion))
        largest = max(len(ld) * len(rd), len(ld) * len(rows), len(rd) * len(cols), len(rows) * len(rd))
        if largest > self.max_block_elements:
            raise MemoryError('determinant transformation block exceeds max_block_elements')
        li, ri = ({d: i for i, d in enumerate(ld)}, {d: i for i, d in enumerate(rd)})
        u, v = (np.zeros((len(ld), len(rows))), np.zeros((len(rd), len(cols))))
        for j, expansion in enumerate(left):
            for d, value in expansion.items():
                u[li[d], j] = value
        for j, expansion in enumerate(right):
            for d, value in expansion.items():
                v[ri[d], j] = value
        matrix = np.array([[self.determinant_element(a, b) for b in rd] for a in ld])
        matrix = matrix.reshape(len(ld), len(rd))
        return u.T @ matrix @ v

class CSFSpinBackend(_CSFIntegralBackend):
    """Native determinant-free Hamiltonian by CG contractions or GUGA segments.

    A CSF is a chain of Clebsch--Gordan tensors, with intermediate spin
    projections as bonds. Local fermionic operator strings, including parity,
    contract pairs of these chains. Cost is polynomial in open-shell count for
    each requested pair. The default GUGA engine uses scalar walk-segment
    factors and a two-state sum over complete intermediate spin walks instead.
    Integral/storage conventions match CSFReferenceBackend exactly.

    ``cache_bytes`` caps retained native coupling payload (0 disables retention).
    ``nthreads`` controls native configuration workers, defaulting to one.
    The per-call tensor budget is shared/divided among active workers; bond and
    connection scratch limits apply per worker. ``prescreen=False`` disables
    the cheap transfer bound but retains prepared configuration bounds during
    screened selection. Rebuild the backend/operator
    after changing integrals. See ``cache_info`` and ``clear_cache``.

    ``coupling_engine="compiled"`` reuses projected signed local maps and skips
    spin-zero sites. ``"python"`` retains the original fermionic-string loop
    as a validation path. Both use the same CG phases and term-sum order.
    ``"guga"`` uses spin-free generators with an explicit phase conversion to
    the same CSF convention; floating-point term ordering differs from CG.
    Fixed-occupation scalars and equal generators are folded before evaluation.
    ``max_transfer_bounds`` independently caps the integral-transfer bound table
    at nbasis**4 doubles (0 disables it); larger tables use direct bounds.

    The default ``selection_engine="ordered"`` uses sorted
    certified double-transfer bounds; ``matrix_index="trie"``, ``"residue"``
    or ``"zobrist"`` indexes only selected occupations. Zobrist hashes update
    removal residues by XOR and verify full endpoint occupations after lookup.
    ``"reference"`` retains the Section 21
    search algorithms. ``max_ordered_bytes`` caps retained sorted lists and
    ``max_index_bytes`` caps each transient matrix index; an insufficient cap
    falls back to reference search. These caps are separate from cache_bytes.

    For GUGA, ``guga_factors="table"`` uses a fixed shared 16,448-byte table
    (analytic fallback above twice-spin 256). ``"spectator"`` additionally skips
    identical empty/closed runs; its two metadata arrays count against the
    tensor-entry budget. ``"reference"`` retains analytic site evaluation.
    ``collect_reuse`` records bounded reduced-pattern histograms and spin-key
    frequencies. ``coupling_reuse`` enables the worker-local
    spin-only matrix cache, capped in total by ``max_reuse_bytes``. It is rebuilt
    per matrix call and never changes HCI selection. Histogram timings include
    instrumentation and must be measured separately from ordinary runs.
    ``reuse_strategy="packed"`` uses exact fixed-size keys and a bounded
    direct-mapped table; ``"string"`` preserves the original reuse probe.
    ``walk_engine="connected"`` enables bounded deterministic structural GUGA
    neighbor generation during selection; ``"adaptive"`` gates it by spin-path
    and channel counts. ``"counted"`` retains exhaustive paths with equivalent
    logical candidate accounting and is the GUGA default; ``max_walk_bytes`` caps
    graph/union scratch across workers with exhaustive fallback.
    ``parallel_engine="shared"`` pipelines source occupations without per-source
    team barriers and tiles large CSR occupation blocks by row. Target-owned
    publication preserves labels while scheduling can change work counters.
    ``"reference"`` retains the preceding scheduler. The shared route bulk
    appends labels already validated by native selection.

    GUGA defaults to spectator factors, counted walks, packed spin reuse, shared
    scheduling, residue matrix indexing and no retained connection cache. When
    an explicit CG engine is selected, unspecified GUGA-only options resolve to
    compatible reference choices. The determinant oracle remains available as
    CSFReferenceBackend.
    """

    def __init__(self, ham, *, max_block_elements=1000000, max_bond_elements=1000000, max_tensor_entries=1000000, max_connection_terms=1000000, cache_bytes=0, nthreads=1, prescreen=True, coupling_engine='guga', max_transfer_bounds=1000000, selection_engine='ordered', max_ordered_bytes=16 * 1024 * 1024, matrix_index='residue', max_index_bytes=64 * 1024 * 1024, collect_reuse=False, max_reuse_patterns=10000, guga_factors=None, coupling_reuse=None, max_reuse_bytes=16 * 1024 * 1024, reuse_strategy='packed', walk_engine=None, max_walk_bytes=4 * 1024 * 1024, parallel_engine='shared', matrix_parallel_engine=None, matrix_kernel='general', max_reuse_bytes_per_worker=None):
        super().__init__(ham, max_block_elements=max_block_elements)
        self.max_bond_elements = _limit(max_bond_elements, 'max_bond_elements')
        self.max_tensor_entries = _limit(max_tensor_entries, 'max_tensor_entries')
        self.max_connection_terms = _limit(max_connection_terms, 'max_connection_terms')
        if self.max_bond_elements < 1 or self.max_tensor_entries < 1:
            raise ValueError('spin tensor allocation limits must be positive')
        if self.max_connection_terms < 0:
            raise ValueError('max_connection_terms must be nonnegative')
        cache_bytes = _integer(cache_bytes, 'cache_bytes')
        max_transfer_bounds = _integer(max_transfer_bounds, 'max_transfer_bounds')
        if max_transfer_bounds < 0:
            raise ValueError('max_transfer_bounds must be nonnegative')
        if parallel_engine not in ('reference', 'shared'):
            raise ValueError('parallel_engine must be reference or shared')
        self.parallel_engine = parallel_engine
        if matrix_parallel_engine is None:
            matrix_parallel_engine = parallel_engine
        if matrix_parallel_engine not in ('reference', 'shared', 'private'):
            raise ValueError('matrix_parallel_engine must be reference, shared, or private')
        self.matrix_parallel_engine = matrix_parallel_engine
        self.nthreads = _integer(nthreads, 'nthreads')
        self.prescreen = bool(prescreen)
        if coupling_engine not in ('compiled', 'python', 'guga'):
            raise ValueError('coupling_engine must be compiled, python, or guga')
        self.coupling_engine = coupling_engine
        if matrix_kernel not in ('general', 'ranked'):
            raise ValueError('matrix_kernel must be general or ranked')
        if matrix_kernel == 'ranked' and coupling_engine != 'guga':
            raise ValueError('ranked matrix_kernel requires GUGA')
        self.matrix_kernel = matrix_kernel
        if guga_factors is None:
            guga_factors = 'spectator' if coupling_engine == 'guga' else 'reference'
        if coupling_reuse is None:
            coupling_reuse = coupling_engine == 'guga'
        if walk_engine is None:
            walk_engine = 'counted' if coupling_engine == 'guga' else 'reference'
        if guga_factors not in ('reference', 'table', 'spectator'):
            raise ValueError('guga_factors must be reference, table, or spectator')
        if coupling_engine != 'guga' and guga_factors != 'reference':
            raise ValueError("guga_factors requires coupling_engine='guga'")
        self.guga_factors = guga_factors
        if selection_engine not in ('reference', 'ordered'):
            raise ValueError('selection_engine must be reference or ordered')
        self.selection_engine = selection_engine
        self._max_ordered_bytes = _integer(max_ordered_bytes, 'max_ordered_bytes')
        if self.max_ordered_bytes < 0:
            raise ValueError('max_ordered_bytes must be nonnegative')
        if matrix_index not in ('reference', 'trie', 'residue', 'zobrist'):
            raise ValueError('matrix_index must be reference, trie, residue, or zobrist')
        self.matrix_index = matrix_index
        self.max_index_bytes = _integer(max_index_bytes, 'max_index_bytes')
        self.max_reuse_patterns = _integer(max_reuse_patterns, 'max_reuse_patterns')
        self.collect_reuse = bool(collect_reuse)
        self.coupling_reuse = bool(coupling_reuse)
        if reuse_strategy not in ('string', 'packed'):
            raise ValueError('reuse_strategy must be string or packed')
        self.reuse_strategy = reuse_strategy
        if walk_engine not in ('reference', 'connected', 'adaptive', 'counted'):
            raise ValueError('walk_engine must be reference, connected, adaptive, or counted')
        if walk_engine != 'reference' and coupling_engine != 'guga':
            raise ValueError("connected walks require coupling_engine='guga'")
        self.walk_engine = walk_engine
        self.max_walk_bytes = _integer(max_walk_bytes, 'max_walk_bytes')
        if self.max_walk_bytes < 0:
            raise ValueError('max_walk_bytes must be nonnegative')
        self.max_reuse_bytes_per_worker = None if max_reuse_bytes_per_worker is None else _integer(max_reuse_bytes_per_worker, 'max_reuse_bytes_per_worker')
        if self.max_reuse_bytes_per_worker is not None and self.max_reuse_bytes_per_worker < 0:
            raise ValueError('max_reuse_bytes_per_worker must be nonnegative')
        self.max_reuse_bytes = _integer(max_reuse_bytes, 'max_reuse_bytes') if self.max_reuse_bytes_per_worker is None else self.nthreads * self.max_reuse_bytes_per_worker
        if self.max_reuse_bytes < 0:
            raise ValueError('max_reuse_bytes must be nonnegative')
        if coupling_engine != 'guga' and self.coupling_reuse:
            raise ValueError("coupling_reuse requires coupling_engine='guga'")
        if self.max_index_bytes < 0 or self.max_reuse_patterns < 0:
            raise ValueError('index and reuse limits must be nonnegative')
        if cache_bytes < 0:
            raise ValueError('cache_bytes must be nonnegative')
        if not 1 <= self.nthreads <= 256:
            raise ValueError('nthreads must be between 1 and 256')
        try:
            from scsf import _csf
        except ImportError as exc:
            raise ImportError('build the native CSF kernel with python setup.py build_ext --inplace PYTHON=/path/to/python') from exc
        self._kernel = _csf.hamiltonian_block
        self._native = _csf
        self._cache = _csf.ConnectionCache(self.one, self.two, cache_bytes, coupling_engine == 'guga', max_transfer_bounds)
        self.last_selection_stats = {}
        self.last_matrix_stats = {}

    def clear_cache(self):
        """Discard prepared couplings/counters; retain integral search tables."""
        self._cache.clear()

    @property
    def cache_info(self):
        """Accounted retained bytes, byte cap, entries, hits, misses, evictions."""
        return dict(self._cache.info())

    @property
    def transfer_bound_info(self):
        """Separate bounded integral-transfer table (entries, bytes, entry cap)."""
        return dict(self._cache.transfer_info())

    @property
    def search_info(self):
        """Bounded sorted transfer-list storage, separate from other caches."""
        return dict(self._cache.search_info())

    @property
    def max_ordered_bytes(self):
        """Sorted-table byte cap fixed for this backend's lifetime."""
        return self._max_ordered_bytes

    @property
    def _guga_mode(self):
        return ('reference', 'table', 'spectator').index(self.guga_factors)

    def block(self, rows, cols, two_s):
        """Contract the requested labels without generating determinant supports."""
        rows, cols = self._validate_block(rows, cols, two_s)
        if not rows or not cols:
            return np.zeros((len(rows), len(cols)))
        guga = self.coupling_engine == 'guga'
        entries, left_max, right_max = (0, [0] * self.nbasis, [0] * self.nbasis)
        for labels, maxima in ((rows, left_max), (cols, right_max)):
            for label in labels:
                spin = 0
                for p, step in enumerate(label):
                    entries += 1 if guga else (spin + 1) * (2 if step in 'ud' else 1)
                    spin += (step == 'u') - (step == 'd')
                    maxima[p] = max(maxima[p], spin + 1)
                if guga:
                    entries += 1
        if guga and self._guga_mode == 2:
            entries *= 3
        if entries > self.max_tensor_entries:
            raise MemoryError('spin tensors exceed max_tensor_entries')
        if guga and self.nbasis and (self.max_bond_elements < 2) or (not guga and any((a * b > self.max_bond_elements for a, b in zip(left_max, right_max)))):
            raise MemoryError('spin contraction exceeds max_bond_elements')
        try:
            return self._kernel(self.one, self.two, rows, cols, two_s, self.max_bond_elements, self.max_tensor_entries, self.max_connection_terms, self._cache, self.coupling_engine == 'compiled', guga, self._guga_mode)
        except ValueError as error:
            if 'exceed' in str(error):
                raise MemoryError(str(error)) from error
            raise

    def generator_block(self, rows, cols, two_s, p, q):
        """Signed GUGA matrix of E[p,q] = sum_spin a†[p,spin] a[q,spin]."""
        return self._guga_operator_block(rows, cols, two_s, (p, q))

    def two_body_block(self, rows, cols, two_s, p, q, r, s):
        """Signed matrix of E[p,q] E[r,s] - delta[q,r] E[p,s].

        The intermediate resolution includes all allowed walks, independently
        of the supplied selected rows/columns. No determinant expansion is used.
        """
        return self._guga_operator_block(rows, cols, two_s, (p, q, r, s))

    def _guga_operator_block(self, rows, cols, two_s, indices):
        rows, cols = self._validate_block(rows, cols, two_s)
        indices = tuple((_integer(p, 'orbital index') for p in indices))
        try:
            return self._native.guga_operator_block(rows, cols, self.nbasis, two_s, indices, self.max_block_elements, self.max_tensor_entries, self.max_bond_elements, self._guga_mode)
        except ValueError as error:
            if 'exceed' in str(error):
                raise MemoryError(str(error)) from error
            raise

    def connected_configurations(self, occupations, *, max_configurations=100000):
        """Return sorted native zero/one/two-transfer occupation endpoints."""
        occ = tuple((_integer(n, 'occupation') for n in occupations))
        if len(occ) != self.nbasis:
            raise ValueError('occupation length differs from orbital count')
        maximum = _limit(max_configurations, 'max_configurations')
        if maximum < 1:
            raise MemoryError('configuration search exceeds max_configurations')
        return [tuple(c) for c in self._native.connected_configurations(occ, maximum)]

    def configuration_bound(self, bra, ket):
        """Bound every CSF coupling of two configurations by operator norms.

        CG combines equal projected local actions with interval rounding.
        GUGA folds occupation scalars and uses projected local operator norms.
        Both cover all spin paths,
        including missing paths, and include a floating-point error envelope.
        """
        bra = tuple((_integer(n, 'occupation') for n in bra))
        ket = tuple((_integer(n, 'occupation') for n in ket))
        return self._native.configuration_bound(self.one, self.two, bra, ket, self.max_connection_terms, self._cache, self.coupling_engine == 'guga')

    def _select_hci(self, wfn, coeffs, eps, *, screening, max_candidates, max_csfs, max_configurations, selection_rule='hci', parent_fraction=1.0):
        selection = self._native.select_hci(self.one, self.two, tuple(wfn), wfn.two_s, coeffs, eps, screening, max_candidates, max_csfs, max_configurations, self.max_bond_elements, self.max_tensor_entries, self.max_connection_terms, self._cache, self.nthreads, self.prescreen, self.coupling_engine == 'compiled', self.coupling_engine == 'guga', self.selection_engine == 'ordered', self.max_ordered_bytes, self._guga_mode, ('reference', 'connected', 'adaptive', 'counted').index(self.walk_engine), self.max_walk_bytes, ('hci', 'configuration', 'bound', 'transfer').index(selection_rule), parent_fraction, self.parallel_engine == 'shared', self.parallel_engine == 'shared')
        if self.parallel_engine == 'shared':
            labels, stats, groups = selection
        else:
            labels, stats = selection
            groups = None
        return (labels, dict(stats), groups)

    def _build_csr(self, wfn, nrow, ncol, symmetric, start, max_entries, matrix_cutoff=1e-10):
        data, indices, indptr, stats = self._native.build_csr(self.one, self.two, tuple(wfn), wfn.two_s, nrow, ncol, symmetric, start, max_entries, self.max_bond_elements, self.max_tensor_entries, self.max_connection_terms, self._cache, self.nthreads, self.coupling_engine == 'compiled', self.coupling_engine == 'guga', ('reference', 'trie', 'residue', 'zobrist').index(self.matrix_index), self.max_index_bytes, self.collect_reuse, self.max_reuse_patterns, self._guga_mode, self.coupling_reuse, self.max_reuse_bytes, self.reuse_strategy == 'packed', self.matrix_parallel_engine != 'reference', self.matrix_parallel_engine == 'private', matrix_cutoff, self.matrix_kernel == 'ranked', _limit(self.max_reuse_bytes_per_worker, 'max_reuse_bytes_per_worker'))
        return (data, indices, indptr, dict(stats))

def _one_electron_inputs(one, wfn, coeffs, nthreads, triplet):
    if not isinstance(wfn, csf_wfn):
        raise TypeError('one-electron CSF operations require csf_wfn')
    if triplet:
        raise NotImplementedError('triplet one-electron CSF operators are not implemented')
    from .config import get_num_threads
    threads = get_num_threads() if nthreads is None else _integer(nthreads, 'nthreads')
    if not 1 <= threads <= 256:
        raise ValueError('nthreads must be in 1..256')
    one, coeffs = (np.asarray(one), np.asarray(coeffs))
    if one.shape != (wfn.nbasis, wfn.nbasis):
        raise ValueError('one-electron operator shape must match the CSF orbital count')
    if one.dtype.kind not in 'biufc' or coeffs.dtype.kind not in 'biufc' or (not np.all(np.isfinite(one))) or (not np.all(np.isfinite(coeffs))):
        raise ValueError('one-electron operator and coefficients must be finite numeric arrays')
    return (one, coeffs, threads)

def csf_one_electron_product(one, wfn, coeffs, *, nthreads=None, triplet=False, max_tensor_entries=None):
    """Apply sum[p,q] one[p,q] E[p,q] in the selected CSF space.

    This spin-scalar (singlet) operator preserves total spin. Orbital matrices
    can be nonsymmetric and complex. Coefficients may be a vector or column
    block, including a prefix of the selected basis (zero-padded implicitly).
    Output has the full selected row dimension. No CSFs are added, no scalar
    core term is included, and no four-index tensor/operator CSR is allocated.
    """
    one, coeffs, threads = _one_electron_inputs(one, wfn, coeffs, nthreads, triplet)
    vector = coeffs.ndim == 1
    if coeffs.ndim not in (1, 2) or coeffs.shape[0] > len(wfn):
        raise ValueError('coefficients must be a selected-basis prefix vector or column block')
    from scsf import _csf
    block = coeffs[:, None] if vector else coeffs
    try:
        result = _csf.one_electron_product(np.ascontiguousarray(one, dtype=complex), tuple(wfn), wfn.two_s, np.ascontiguousarray(block, dtype=complex), threads, _limit(max_tensor_entries, 'max_tensor_entries'))
    except ValueError as error:
        if 'exceed' in str(error):
            raise MemoryError(str(error)) from error
        raise
    if not np.iscomplexobj(one) and (not np.iscomplexobj(coeffs)):
        result = result.real.copy()
    return result[:, 0] if vector else result

def add_csf_one_electron(one, wfn, coeffs, eps=1e-05, *, nthreads=None, triplet=False, max_csfs=None, max_tensor_entries=None, return_stats=False):
    """Append missing CSFs with max_J abs(O[I,J]*c[J]) > eps.

    Only zero/one-electron transfers can couple. Parents are immutable within
    a call, additions are sorted/deduplicated, and publication is transactional.
    Nonsymmetric/complex singlet orbital matrices and signed/complex parent
    vectors are supported. ``max_csfs`` guards the final selected basis size.
    """
    one, coeffs, threads = _one_electron_inputs(one, wfn, coeffs, nthreads, triplet)
    if coeffs.shape != (len(wfn),):
        raise ValueError('selection needs one coefficient per original CSF')
    if not np.isfinite(eps) or eps < 0:
        raise ValueError('eps must be finite and nonnegative')
    maximum = _limit(max_csfs, 'max_csfs')
    if maximum < len(wfn):
        raise ValueError('max_csfs must be at least the original CSF count')
    from scsf import _csf
    try:
        labels, stats, groups = _csf.select_one_electron(np.ascontiguousarray(one, dtype=complex), tuple(wfn), wfn.two_s, np.ascontiguousarray(np.abs(coeffs), dtype=float), eps, threads, _limit(max_tensor_entries, 'max_tensor_entries'), maximum - len(wfn))
    except ValueError as error:
        if 'exceed' in str(error):
            raise MemoryError(str(error)) from error
        raise
    wfn._append_native(labels, groups)
    return (len(labels), dict(stats)) if return_stats else len(labels)

def _connected(a, b):
    return sum((abs(x - y) for x, y in zip(a, b))) <= 4

class csf_op:
    """CSF Hamiltonian with native sparse products and pluggable couplings.

    Stored values, matvec, matmat, and diagonal are electronic. The example
    driver uses external Davidson diagonalization. Symmetric storage is lower-triangular.
    Prefix rectangular matrices require symmetric=False. Use explicit csf_op
    using native spin-adapted matrix construction and products.
    ``product_engine="scipy"`` preserves the original product implementation.
    Product threads default to the coupling backend's worker count (one for the
    reference backend); ``product_nthreads`` overrides them independently.
    An omitted backend selects CSFSpinBackend with its measured defaults.
    ``product_mode`` selects the custom lower-CSR ``native`` traversal
    (default), the full-CSR ``native-gather`` traversal. There is no MKL dependency.
    """

    def __init__(self, ham, wfn, nrow=-1, ncol=-1, symmetric=True, *, backend=None, max_matrix_entries=2000000, product_engine='native', product_nthreads=None, product_mode='native', matrix_cutoff=1e-10, storage_engine='auto'):
        if not isinstance(wfn, csf_wfn):
            raise TypeError('csf_op requires csf_wfn')
        if ham.nbasis != wfn.nbasis:
            raise ValueError('Hamiltonian and CSF orbital counts differ')
        self.backend = backend if backend is not None else CSFSpinBackend(ham)
        if self.backend.nbasis != wfn.nbasis or self.backend._ham is not ham:
            raise ValueError('backend must use the supplied Hamiltonian')
        self._ham = ham
        self._sector = (wfn.nbasis, wfn.nelec, wfn.two_s)
        self._labels = tuple(wfn)
        self._rectangular = nrow != -1 or ncol != -1
        nrow = len(wfn) if nrow == -1 else _integer(nrow, 'nrow')
        ncol = len(wfn) if ncol == -1 else _integer(ncol, 'ncol')
        if not (0 <= nrow <= len(wfn) and 0 <= ncol <= len(wfn)):
            raise ValueError('matrix dimensions must be prefixes of the CSF basis')
        if symmetric and nrow != ncol:
            raise ValueError('symmetric CSF storage requires square dimensions')
        self.symmetric = bool(symmetric)
        if product_engine not in ('native', 'scipy'):
            raise ValueError("product_engine must be 'native' or 'scipy'")
        self._product_engine = product_engine
        if product_mode not in ('native', 'native-gather'):
            raise ValueError('product_mode must be native or native-gather')
        self._product_mode = product_mode
        self.matrix_cutoff = float(matrix_cutoff)
        if not np.isfinite(self.matrix_cutoff) or self.matrix_cutoff < 0:
            raise ValueError('matrix_cutoff must be finite and nonnegative')
        if storage_engine not in ('auto', 'native', 'snapshot'):
            raise ValueError('storage_engine must be auto, native, or snapshot')
        appendable = self.symmetric and product_engine == 'native' and (product_mode == 'native')
        self._storage_engine = ('native' if appendable else 'snapshot') if storage_engine == 'auto' else storage_engine
        if self._storage_engine == 'native' and (not appendable):
            raise ValueError("native CSR storage requires symmetric product_mode='native' and product_engine='native'")
        self._product_nthreads = _integer(getattr(self.backend, 'nthreads', 1) if product_nthreads is None else product_nthreads, 'product_nthreads')
        if not 1 <= self._product_nthreads <= 256:
            raise ValueError('product_nthreads must be between 1 and 256')
        self.ecore = self.backend.ecore
        self.max_matrix_entries = _limit(max_matrix_entries, 'max_matrix_entries')
        self._matrix = None
        if self.storage_engine == 'native':
            from scsf import _csf
            data, indices, indptr, stats = self._build_rows(wfn, nrow, ncol)
            self._products = _csf._CSRProducts(data, indices, indptr, nrow, ncol, self.symmetric, self.product_nthreads, self.product_mode)
        else:
            matrix, stats = self._build(wfn, nrow, ncol)
            products = self._prepare_products(matrix)
            self._matrix, self._products = (matrix, products)
        self._matrix_cutoff = self.matrix_cutoff
        if stats is not None:
            self.backend.last_matrix_stats = stats
    shape = property(lambda self: self._products.shape if self.storage_engine == 'native' else self._matrix.shape)
    size = property(lambda self: self._products.nnz if self.storage_engine == 'native' else self._matrix.nnz)
    dtype = property(lambda self: np.dtype(float))
    storage_engine = property(lambda self: self._storage_engine)
    product_engine = property(lambda self: self._product_engine)
    product_nthreads = property(lambda self: self._product_nthreads)
    product_mode = property(lambda self: self._product_mode)

    @property
    def product_info(self):
        """Product/storage settings and owned native CSR payload (excludes scratch)."""
        return dict(engine=self.product_engine, nthreads=self.product_nthreads, mode=self.product_mode, storage_engine=self.storage_engine, matrix_cutoff=self.matrix_cutoff, active_threads=self._products.active_threads if self._products is not None else None, csr_bytes=self._products.storage_bytes if self._products is not None else 0, output_scratch_bytes_per_column=self._products.output_scratch_bytes_per_column if self._products is not None else 0, **dict(self._products.storage_info) if self._products is not None else {})

    def _prepare_products(self, matrix):
        if self.product_engine == 'scipy':
            return None
        from scsf import _csf
        return _csf._CSRProducts(matrix.data, matrix.indices, matrix.indptr, *matrix.shape, self.symmetric, self.product_nthreads, self.product_mode)

    def _build(self, wfn, nrow, ncol, start=0):
        data, indices, indptr, stats = self._build_rows(wfn, nrow, ncol, start)
        if start:
            data = np.concatenate((self._matrix.data, data))
            indices = np.concatenate((self._matrix.indices, indices))
            indptr = np.concatenate((self._matrix.indptr, indptr[1:] + self._matrix.nnz))
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(nrow, ncol))
        return (matrix, stats)

    def _build_rows(self, wfn, nrow, ncol, start=0):
        """Build only the new row tail, with local row pointers starting at zero."""
        old_size = self.size if start else 0
        remaining = self.max_matrix_entries - old_size
        if remaining < 0:
            raise MemoryError('CSR matrix exceeds max_matrix_entries')
        if isinstance(self.backend, CSFSpinBackend):
            return self.backend._build_csr(wfn, nrow, ncol, self.symmetric, start, remaining, self.matrix_cutoff)
        rows, cols, data = ([], [], [])
        for a, all_rows in wfn._configs.items():
            ia = [i for i in all_rows if start <= i < nrow]
            if not ia:
                continue
            for b, all_cols in wfn._configs.items():
                ib = [i for i in all_cols if i < ncol and (not self.symmetric or i <= max(ia))]
                if not ib or not _connected(a, b):
                    continue
                block = self.backend.block([wfn[i] for i in ia], [wfn[i] for i in ib], wfn.two_s)
                for r, i in enumerate(ia):
                    for c, j in enumerate(ib):
                        if self.symmetric and j > i:
                            continue
                        value = block[r, c]
                        if abs(value) > self.matrix_cutoff or i == j:
                            if len(data) >= remaining:
                                raise MemoryError('CSR matrix exceeds max_matrix_entries')
                            rows.append(i - start)
                            cols.append(j)
                            data.append(value)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(nrow - start, ncol), dtype=float)
        matrix.sort_indices()
        return (matrix.data, matrix.indices, matrix.indptr, None)

    def update(self, ham, wfn):
        """Append new lower rows; full storage rebuilds to include new columns.

        Integrals are a snapshot. Pass the original ham object and an unchanged
        basis prefix; changed integrals or rectangular prefixes need a new op.
        """
        if self._rectangular:
            raise ValueError('rebuild prefix/rectangular operators explicitly')
        if self.matrix_cutoff != self._matrix_cutoff:
            raise ValueError('rebuild the operator to change matrix_cutoff')
        if ham is not self._ham or (wfn.nbasis, wfn.nelec, wfn.two_s) != self._sector or tuple(wfn)[:len(self._labels)] != self._labels:
            raise ValueError('update requires the original Hamiltonian and unchanged CSF prefix')
        if len(wfn) == len(self._labels):
            return
        labels = tuple(wfn)
        if self.storage_engine == 'native':
            data, indices, indptr, stats = self._build_rows(wfn, len(wfn), len(wfn), len(self._labels))
            self._products.append_rows(data, indices, indptr, len(wfn))
            self._labels = labels
        else:
            matrix, stats = self._build(wfn, len(wfn), len(wfn), len(self._labels) if self.symmetric else 0)
            products = self._prepare_products(matrix)
            self._matrix, self._labels, self._products = (matrix, labels, products)
        if stats is not None:
            self.backend.last_matrix_stats = stats

    def to_scipy(self):
        """Return a full electronic CSR matrix copy, reflecting its triangle."""
        matrix = sparse.csr_matrix((self.data(), self.indices(), self.indptr()), shape=self.shape) if self.storage_engine == 'native' else self._matrix
        if self.symmetric:
            return matrix + sparse.tril(matrix, k=-1).T
        return matrix.copy()

    def data(self):
        return self._products.data() if self.storage_engine == 'native' else self._matrix.data.copy()

    def indices(self):
        return self._products.indices() if self.storage_engine == 'native' else self._matrix.indices.copy()

    def indptr(self):
        return self._products.indptr() if self.storage_engine == 'native' else self._matrix.indptr.copy()

    def diagonal(self):
        return self._products.diagonal() if self.storage_engine == 'native' else self._matrix.diagonal().copy()

    def get_element(self, i, j):
        i, j = (_integer(i, 'row'), _integer(j, 'column'))
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise IndexError('CSF matrix index out of bounds')
        if self.storage_engine == 'native':
            return self._products.get_element(i, j)
        if self.symmetric and i < j:
            i, j = (j, i)
        return float(self._matrix[i, j])

    def matvec(self, x, out=None):
        x = _real_vector(x, self.shape[1])
        if self._products is not None:
            result = self._products.matvec(x)
        else:
            result = self._matrix @ x
            if self.symmetric:
                result += sparse.tril(self._matrix, k=-1).T @ x
        if out is not None:
            if not isinstance(out, np.ndarray) or out.shape != result.shape or out.dtype != np.float64:
                raise ValueError('out must be a float64 array of the output shape')
            out[:] = result
            return out
        return result
    __call__ = matvec

    def matmat(self, x):
        """Apply the electronic Hamiltonian to a block of column vectors."""
        if np.iscomplexobj(x):
            raise TypeError('CSF matmat supports real coefficients only')
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] != self.shape[1] or (not np.all(np.isfinite(x))):
            raise ValueError('trial block must be finite with shape (ncol, nvector)')
        if self._products is not None:
            return self._products.matmat(x)
        result = self._matrix @ x
        if self.symmetric:
            result += sparse.tril(self._matrix, k=-1).T @ x
        return result


def connected_configurations(occupations, *, max_configurations=100000):
    """Yield unique endpoints of zero, one, or two electron occupation transfers.

    Check the endpoint-set budget during construction, before path enumeration.
    """
    max_configurations = _limit(max_configurations, 'max_configurations')
    if max_configurations < 1:
        raise MemoryError('configuration search exceeds max_configurations')
    occupations = tuple((_integer(v, 'occupation') for v in occupations))
    if any((v not in (0, 1, 2) for v in occupations)):
        raise ValueError('occupations must be 0, 1, or 2')
    targets = {occupations}
    occupied = [p for p, n in enumerate(occupations) if n]
    for order in (1, 2):
        for holes in combinations_with_replacement(occupied, order):
            intermediate = list(occupations)
            for p in holes:
                intermediate[p] -= 1
            if min(intermediate) < 0:
                continue
            available = [p for p, n in enumerate(intermediate) if n < 2]
            for particles in combinations_with_replacement(available, order):
                target = intermediate.copy()
                for p in particles:
                    target[p] += 1
                if max(target) <= 2:
                    targets.add(tuple(target))
                    if len(targets) > max_configurations:
                        raise MemoryError('configuration search exceeds max_configurations')
    yield from sorted(targets)

def _selection_parents(wfn, coeffs, fraction):
    """Retain entire parent occupations by descending squared coefficient norm."""
    scale = np.max(np.abs(coeffs), initial=0.0)
    weighted = []
    for config, indices in wfn._configs.items():
        weight = float(np.sum((coeffs[indices] / scale) ** 2)) if scale else 0.0
        if weight:
            weighted.append((weight, config, indices))
    weighted.sort(key=lambda item: (-item[0], item[1]))
    total = sum((item[0] for item in weighted))
    retained, count = (0.0, 0)
    filtered = np.zeros_like(coeffs)
    for weight, _, indices in weighted:
        if retained >= fraction * total:
            break
        filtered[indices] = coeffs[indices]
        retained += weight
        count += 1
    stats = dict(parent_fraction=fraction, parent_configurations=len(weighted), retained_parent_configurations=count, retained_parent_csfs=int(np.count_nonzero(filtered)), retained_parent_weight=retained / total if total else 1.0)
    return (filtered, stats)

def add_csf_hci(ham, wfn, coeffs, eps=1e-05, *, backend=None, max_candidates=100000, max_csfs=100000, max_configurations=100000, implementation='auto', screening=True, return_stats=False, selection_rule='hci', parent_fraction=1.0):
    """Append CSF candidates using strict HCI or opt-in configuration rules.

    By default select iff max_J abs(H_IJ*c_J) > eps, with immutable parents and deterministic
    label ordering. Includes missing paths of selected configurations. This
    native search uses a conservative configuration bound when screening=True.
    implementation="python" retains the unscreened oracle; "auto" chooses
    native SCI loops for CSFSpinBackend and Python for the determinant reference.
    An omitted backend selects native GUGA; pass CSFReferenceBackend explicitly
    for the determinant oracle and reuse op.backend in iterative calculations.
    Coefficients can be the maximum absolute coefficient over same-spin roots.
    Candidate and allocation limits raise before mutating wfn.
    ``selection_rule="configuration"`` fills every coupling of an occupation
    after one strict witness. ``"bound"`` and ``"transfer"`` admit complete
    occupations using prepared or coarse certified bounds, without evaluating
    individual CSF couplings. They can overselect and require native selection.
    For fixed parents and epsilon these rules include the strict HCI result;
    Hamiltonian construction and diagonalization remain unmodified. Work limits
    accept None to disable them up to the native index capacity.
    ``parent_fraction < 1`` restricts generation to the largest occupation norms
    covering that fraction of sum(abs(coeffs)**2). All original CSFs remain in
    the variational matrix. This optional heuristic can miss strict-HCI matches
    from omitted parents; lower epsilon and compare converged energies. It can
    be combined with any selection rule, and never changes the input vector.
    """
    if not isinstance(wfn, csf_wfn):
        raise TypeError('add_csf_hci requires csf_wfn')
    max_candidates = _limit(max_candidates, 'max_candidates')
    max_csfs = _limit(max_csfs, 'max_csfs')
    max_configurations = _limit(max_configurations, 'max_configurations')
    if max_configurations < 1:
        raise MemoryError('configuration search exceeds max_configurations')
    if implementation not in ('auto', 'native', 'python'):
        raise ValueError('implementation must be auto, native, or python')
    if selection_rule not in ('hci', 'configuration', 'bound', 'transfer'):
        raise ValueError('selection_rule must be hci, configuration, bound, or transfer')
    if max_candidates < 0 or max_csfs < len(wfn):
        raise ValueError('require max_candidates >= 0 and max_csfs >= len(wfn)')
    coeffs = _real_vector(coeffs, len(wfn))
    if not np.isfinite(parent_fraction) or not 0 < parent_fraction <= 1:
        raise ValueError('parent_fraction must be finite and in (0, 1]')
    if not np.isfinite(eps) or eps < 0:
        raise ValueError('eps must be finite and nonnegative')
    if ham.nbasis != wfn.nbasis:
        raise ValueError('Hamiltonian and CSF orbital counts differ')
    backend = backend if backend is not None else CSFSpinBackend(ham)
    if backend.nbasis != wfn.nbasis or backend._ham is not ham:
        raise ValueError('backend must use the supplied Hamiltonian')
    native = isinstance(backend, CSFSpinBackend)
    if implementation == 'native' and (not native):
        raise TypeError('native SCI selection requires CSFSpinBackend')
    if selection_rule in ('bound', 'transfer') and (not native or implementation == 'python'):
        raise ValueError('bound/transfer selection requires the native CSFSpinBackend')
    if native and implementation != 'python':
        labels, stats, groups = backend._select_hci(wfn, coeffs, eps, screening=screening, max_candidates=max_candidates, max_csfs=max_csfs, max_configurations=max_configurations, selection_rule=selection_rule, parent_fraction=parent_fraction)
        if groups is not None:
            wfn._append_native(labels, groups)
        else:
            for label in labels:
                wfn.add_csf(label)
        backend.last_selection_stats = stats
        return (len(labels), dict(stats)) if return_stats else len(labels)
    parent_stats = {}
    if parent_fraction < 1:
        coeffs, parent_stats = _selection_parents(wfn, coeffs, parent_fraction)
    selected, candidates, promoted, accounted = (set(), set(), set(), set())
    logical_candidates = 0
    for source, indices in wfn._configs.items():
        parents = [i for i in indices if coeffs[i] != 0]
        if not parents:
            continue
        for target in connected_configurations(source, max_configurations=max_configurations):
            if target in promoted:
                continue
            if selection_rule == 'configuration' and target not in accounted:
                n = target.count(1)
                k = (n - wfn.two_s) // 2
                dimension = _choose(n, k) - _choose(n, k - 1)
                logical_candidates += dimension - len(wfn._configs.get(target, ()))
                if logical_candidates > max_candidates:
                    raise MemoryError('HCI search exceeds max_candidates')
                accounted.add(target)
            for label in _paths(target, wfn.two_s):
                if label in wfn._indices or label in selected:
                    continue
                candidates.add(label)
                if len(candidates) > max_candidates:
                    raise MemoryError('HCI search exceeds max_candidates')
                block = backend.block((label,), [wfn[i] for i in parents], wfn.two_s)[0]
                if np.any(np.abs(block * coeffs[parents]) > eps):
                    if selection_rule == 'configuration':
                        promoted.add(target)
                        missing = [s for s in _paths(target, wfn.two_s) if s not in wfn._indices]
                        if len(wfn) + len(selected) + len(missing) > max_csfs:
                            raise MemoryError('selected basis exceeds max_csfs')
                        selected.update(missing)
                        break
                    selected.add(label)
    if len(wfn) + len(selected) > max_csfs:
        raise MemoryError('selected basis exceeds max_csfs')
    for label in sorted(selected):
        wfn.add_csf(label)
    stats = dict(added=len(selected), candidates=logical_candidates if selection_rule == 'configuration' else len(candidates))
    stats.update(parent_stats)
    return (len(selected), stats) if return_stats else len(selected)
