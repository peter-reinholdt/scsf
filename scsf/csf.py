# SCSF; licensed under GPL-3.0-or-later.
"""Genealogical CSF spaces and spin-scalar one-electron operators.

Orbitals are coupled in their supplied order. Steps ``0,u,d,2`` denote empty, spin-up-coupled,
spin-down-coupled, and doubly occupied orbitals; ``u/d`` are not alpha/beta spins.
"""
import json
from functools import lru_cache
from math import comb
from operator import index
import numpy as np
from .symmetry import PointGroup, check_symmetry

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

def csf_dimension(nbasis, nelec, two_s, *, symmetry=None, target_irrep=None):
    """Return the exact fixed-spin dimension, using Python's unbounded integers."""
    nbasis, nelec, two_s = _sector(nbasis, nelec, two_s)
    a, b = ((nelec + two_s) // 2, (nelec - two_s) // 2)
    symmetry = check_symmetry(symmetry, nbasis)
    if target_irrep is not None:
        if symmetry is None:
            raise ValueError('target_irrep requires a PointGroup')
        target = symmetry.irrep(target_irrep)
        extent = len(symmetry.names)
        counts = [[0]*extent for _ in range(min(nbasis, a + 1) + 1)]
        counts[0][0] = 1
        for p, label in enumerate(symmetry.orbital_irreps, start=1):
            for electrons in range(min(p, len(counts) - 1), 0, -1):
                for code, count in enumerate(counts[electrons - 1]):
                    counts[electrons][code ^ label] += count

        def determinants(up, down):
            if down < 0 or up >= len(counts):
                return 0
            return sum(count * counts[down][target ^ code] for code, count in enumerate(counts[up]))

        return determinants(a, b) - determinants(a + 1, b - 1)
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

    def __init__(self, nbasis, nelec, two_s, *, symmetry=None, target_irrep=None):
        self._nbasis, self._nelec, self._two_s = _sector(nbasis, nelec, two_s)
        self._symmetry = check_symmetry(symmetry, self.nbasis)
        if target_irrep is not None and self._symmetry is None:
            raise ValueError('target_irrep requires a PointGroup')
        self._target_irrep = None if target_irrep is None else self._symmetry.irrep(target_irrep)
        self._labels = []
        self._indices = {}
        self._configs = {}
        self._occupation_snapshot = None
        self._occupation_symmetry = None
    nbasis = property(lambda self: self._nbasis)
    nelec = property(lambda self: self._nelec)
    two_s = property(lambda self: self._two_s)
    ncsf = property(lambda self: len(self))
    nconfig = property(lambda self: len(self._configs))
    symmetry = property(lambda self: self._symmetry)
    target_irrep = property(lambda self: self._target_irrep)

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return iter(self._labels)

    def __getitem__(self, item):
        return self._labels[item] if not isinstance(item, slice) else tuple(self._labels[item])

    def __copy__(self):
        duplicate = type(self)(self.nbasis, self.nelec, self.two_s, symmetry=self.symmetry,
                               target_irrep=self.target_irrep)
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
        self._validate_irrep(label)
        return label

    def _validate_irrep(self, occupation):
        if self.target_irrep is not None and self.symmetry.configuration_irrep(occupation) != self.target_irrep:
            raise ValueError('configuration lies outside the target irrep')

    def add_csf(self, steps):
        """Append one validated label; return its index (also for duplicates)."""
        label = self._validate(steps)
        if label not in self._indices:
            self._indices[label] = len(self)
            self._configs.setdefault(_configuration(label), []).append(len(self))
            self._labels.append(label)
            self._occupation_snapshot = None
        return self._indices[label]

    def _occupation_basis(self, symmetry=None):
        symmetry = check_symmetry(symmetry, self.nbasis)
        if self.symmetry is not None and symmetry is not None and self.symmetry != symmetry:
            raise ValueError('wavefunction and Hamiltonian orbital irreps must match')
        symmetry = self.symmetry if symmetry is None else symmetry
        if self._occupation_snapshot is None or self._occupation_symmetry != symmetry:
            from . import _csf
            irreps = () if symmetry is None else symmetry.orbital_irreps
            target = -1 if self.target_irrep is None else self.target_irrep
            self._occupation_snapshot = _csf._OccupationBasis(self.nbasis, self.nelec, self.two_s,
                                                               tuple(self), irreps, target)
            self._occupation_symmetry = symmetry
        return self._occupation_snapshot

    def _append_native(self, labels):
        """Publish trusted labels returned by the native property selector."""
        if not labels:
            return
        configurations = self._configs.copy()
        additions = {}
        for row, label in enumerate(labels, start=len(self)):
            self._validate_irrep(label)
            additions.setdefault(_configuration(label), []).append(row)
        for occupation, rows in additions.items():
            configurations[occupation] = self._configs.get(occupation, []) + rows
        indices = self._indices.copy()
        indices.update(zip(labels, range(len(self), len(self) + len(labels))))
        self._labels, self._indices, self._configs = self._labels + labels, indices, configurations
        self._occupation_snapshot = None

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

    def add_configuration(self, occupations):
        """Append all allowed couplings of an occupation; return number added."""
        occ = self._occupations(occupations)
        self._validate_irrep(occ)
        old = len(self)
        for label in _paths(occ, self.two_s):
            self.add_csf(label)
        return len(self) - old

    def _sector_configurations(self, choices=(0, 1, 2)):
        orbitals = self.nbasis
        labels = () if self.symmetry is None else self.symmetry.orbital_irreps

        @lru_cache(maxsize=None)
        def reachable(p, left, open_needed, irrep):
            remaining = orbitals-p
            if left < 0 or left > 2*remaining or open_needed > min(left, 2*remaining-left):
                return False
            if not remaining:
                return left == irrep == open_needed == 0
            return (reachable(p+1, left, open_needed, irrep) or
                    reachable(p+1, left-1, max(0, open_needed-1), irrep ^ labels[p]) or
                    reachable(p+1, left-2, open_needed, irrep))

        def configurations(p, left, occ, open_needed, irrep):
            if self.target_irrep is not None and not reachable(p, left, open_needed, irrep):
                return
            if p == orbitals:
                if left == 0:
                    yield occ
                return
            for n in choices:
                if 0 <= left-n <= 2*(orbitals-p-1):
                    change = labels[p] if labels and n == 1 else 0
                    yield from configurations(p+1, left-n, occ+(n,), max(0, open_needed-(n == 1)), irrep^change)

        yield from configurations(0, self.nelec, (), self.two_s, self.target_irrep or 0)

    def add_all_csfs(self):
        """Enumerate all CSFs in the spin and optional point-group sector."""
        old = len(self)
        for occ in self._sector_configurations():
            self.add_configuration(occ)
        return len(self) - old

    def _reference_in_sector(self, occupations):
        if self.target_irrep is not None and self.symmetry.configuration_irrep(occupations) != self.target_irrep:
            occupations = next(self._sector_configurations((2, 1, 0)), None)
            if occupations is None:
                raise ValueError('target irrep has no spin-compatible configurations')
        return next(_paths(occupations, self.two_s))

    def _fill_sector_seed(self, size, *, complete=False):
        for occupation in self._sector_configurations((2, 1, 0)):
            if len(self) >= size:
                return
            if complete:
                self.add_configuration(occupation)
            else:
                for label in _paths(occupation, self.two_s):
                    self.add_csf(label)
                    if len(self) >= size:
                        return

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
        if self.symmetry is not None:
            payload.update(version=1, symmetry=self.symmetry._encode(), target_irrep=self.target_irrep)
        with open(filename, 'w', encoding='utf8') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')

    @classmethod
    def from_file(cls, filename):
        """Read a basis file containing exactly the sector and ordered labels."""
        with open(filename, encoding='utf8') as stream:
            data = json.load(stream)
        fields = {'nbasis', 'nelec', 'two_s', 'labels'}
        if not isinstance(data, dict) or set(data) not in (fields, fields | {'version', 'symmetry', 'target_irrep'}):
            raise ValueError('basis file must contain nbasis, nelec, two_s and labels')
        symmetry = None
        if 'version' in data:
            if data['version'] != 1:
                raise ValueError('unsupported basis file version')
            symmetry = PointGroup._decode(data['symmetry'])
        wfn = cls(data['nbasis'], data['nelec'], data['two_s'], symmetry=symmetry,
                  target_irrep=data.get('target_irrep'))
        for label in data['labels']:
            if wfn.index_csf(label) != -1:
                raise ValueError('duplicate CSF in basis file')
            wfn.add_csf(label)
        return wfn

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

def csf_one_electron_product(one, wfn, coeffs, *, nthreads=None, triplet=False, symmetry=None):
    """Apply sum[p,q] one[p,q] E[p,q] in the selected CSF space.

    This spin-scalar (singlet) operator preserves total spin. Orbital matrices
    can be nonsymmetric and complex. Coefficients may be a vector or column
    block, including a prefix of the selected basis (zero-padded implicitly).
    The direct product returns one output row per selected CSF and applies
    only the supplied one-electron operator.
    """
    one, coeffs, threads = _one_electron_inputs(one, wfn, coeffs, nthreads, triplet)
    vector = coeffs.ndim == 1
    if coeffs.ndim not in (1, 2) or coeffs.shape[0] > len(wfn):
        raise ValueError('coefficients must be a selected-basis prefix vector or column block')
    from scsf import _csf
    block = coeffs[:, None] if vector else coeffs
    result = _csf.one_electron_product(np.ascontiguousarray(one, dtype=complex), wfn._occupation_basis(symmetry),
                                       np.ascontiguousarray(block, dtype=complex), threads)
    if not np.iscomplexobj(one) and (not np.iscomplexobj(coeffs)):
        result = result.real.copy()
    return result[:, 0] if vector else result

def add_csf_one_electron(one, wfn, coeffs, eps=1e-05, *, nthreads=None, triplet=False,
                         return_stats=False, symmetry=None):
    """Append missing CSFs with max_J abs(O[I,J]*c[J]) > eps.

    Only zero/one-electron transfers can couple. Parents are immutable within
    a call, additions are sorted/deduplicated, and publication is transactional.
    Nonsymmetric/complex singlet orbital matrices and signed/complex parent
    vectors are supported.
    """
    one, coeffs, threads = _one_electron_inputs(one, wfn, coeffs, nthreads, triplet)
    if coeffs.shape != (len(wfn),):
        raise ValueError('selection needs one coefficient per original CSF')
    if not np.isfinite(eps) or eps < 0:
        raise ValueError('eps must be finite and nonnegative')
    from scsf import _csf
    labels, stats = _csf.select_one_electron(np.ascontiguousarray(one, dtype=complex), wfn._occupation_basis(symmetry),
                                             np.ascontiguousarray(np.abs(coeffs), dtype=float), eps, threads)
    wfn._append_native(labels)
    return (len(labels), dict(stats)) if return_stats else len(labels)
