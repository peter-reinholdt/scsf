"""Real spatial integrals in the CSF convention two[p,r,q,s] = (pq|rs)."""
from itertools import accumulate
import numpy as np
from .symmetry import PointGroup, check_symmetry


_ATOL = 1e-12


def _one_electron(ecore, one, symmetry, symmetry_atol=_ATOL):
    if np.iscomplexobj(ecore) or np.iscomplexobj(one):
        raise TypeError('CSF integrals must be real')
    ecore = float(ecore)
    one = np.array(one, dtype=float, order='C', copy=True)
    if one.ndim != 2 or one.shape[0] != one.shape[1]:
        raise ValueError('one-electron integrals must be a square matrix')
    if not np.isfinite(ecore) or not np.all(np.isfinite(one)):
        raise ValueError('integrals must be finite')
    if not np.allclose(one, one.T, rtol=0., atol=_ATOL):
        raise ValueError('one-electron integrals must be symmetric')
    if symmetry is not None:
        labels = np.asarray(check_symmetry(symmetry, len(one)).orbital_irreps, dtype=np.intp)
        forbidden = labels[:, None] != labels
        _check_forbidden(one[forbidden], symmetry_atol)
        one[forbidden] = 0.
    one.flags.writeable = False
    return ecore, one


def _check_forbidden(values, atol=_ATOL):
    if np.any(~np.isfinite(values)) or np.any(np.abs(values) > atol):
        raise ValueError('integrals contain nonzero point-group forbidden elements')


def _validate_dense_blocks(g, symmetry, *, normalize=False):
    labels = np.asarray(symmetry.orbital_irreps, dtype=np.intp)
    pair_irreps = labels[:, None] ^ labels
    for p, row in enumerate(pair_irreps):
        for q, code in enumerate(row):
            forbidden = pair_irreps != code
            _check_forbidden(g[p, q][forbidden])
            if normalize:
                g[p, q][forbidden] = 0.


class Hamiltonian:
    """Own real integral arrays in physicist order.

    Construct with ``Hamiltonian(ecore, one, two)`` where
    ``two[p,r,q,s] = (pq|rs)``. Arrays are copied into contiguous owned storage.
    """
    def __init__(self, ecore, one, two, *, symmetry=None):
        if one is None or two is None:
            raise TypeError('provide ecore, one and two arrays')
        if any(np.iscomplexobj(value) for value in (ecore, one, two)):
            raise TypeError('CSF integrals must be real')
        self.ecore, self.one_mo = _one_electron(ecore, one, symmetry)
        self.two_mo = np.array(two, dtype=float, order='C', copy=True)
        self.nbasis = len(self.one_mo)
        self._symmetry = symmetry
        if self.two_mo.shape != (self.nbasis,)*4:
            raise ValueError('two-electron integrals must have shape (nbasis,)*4')
        if not np.all(np.isfinite(self.two_mo)):
            raise ValueError('integrals must be finite')
        g = _chemist_symmetric(self)
        if symmetry is not None:
            _validate_dense_blocks(g, symmetry, normalize=True)
        self.two_mo.flags.writeable = False

    symmetry = property(lambda self: self._symmetry)


hamiltonian = Hamiltonian


def _chemist_symmetric(ham, atol=1e-12):
    g = ham.two_mo.transpose(0, 2, 1, 3)
    if not all(np.allclose(g, other, rtol=0., atol=atol)
               for other in (g.swapaxes(0, 1), g.swapaxes(2, 3), g.transpose(2, 3, 0, 1))):
        raise ValueError('ERI tensor must have real spin-free permutation symmetry')
    return g


class PackedHamiltonian:
    """Exact eightfold-packed real ERIs, with both triangles in NumPy order."""

    _block_packed = False

    def __init__(self, ecore, one, packed, *, symmetry=None):
        self.ecore, self.one_mo = _one_electron(ecore, one, symmetry)
        self.nbasis = len(self.one_mo)
        self._symmetry = symmetry
        if np.iscomplexobj(packed):
            raise TypeError('CSF integrals must be real')
        pairs = self.nbasis*(self.nbasis+1)//2
        self.packed_eri = np.array(packed, dtype=float, order='C', copy=True)
        blocks = symmetry._pair_blocks()[2] if symmetry is not None else None
        sizes = (pairs,) if not self._block_packed else tuple(map(len, blocks))
        self.block_offsets = (0, *accumulate(n*(n+1)//2 for n in sizes))
        if self.block_offsets[-1] > np.iinfo(np.intp).max:
            raise OverflowError('integral dimensions exceed index capacity')
        if self.packed_eri.shape != (self.block_offsets[-1],):
            raise ValueError('packed ERIs must be the lower triangle of the orbital-pair matrix')
        if not np.all(np.isfinite(self.packed_eri)):
            raise ValueError('integrals must be finite')
        if blocks is not None and not self._block_packed and sum(bool(len(b)) for b in blocks) > 1:
            labels = np.empty(pairs, dtype=np.intp)
            for code, block in enumerate(blocks):
                labels[block] = code
            for i, code in enumerate(labels):
                begin = i*(i+1)//2
                row = self.packed_eri[begin:begin+i+1]
                forbidden = labels[:i+1] != code
                _check_forbidden(row[forbidden])
                row[forbidden] = 0.
        self.packed_eri.flags.writeable = False

    symmetry = property(lambda self: self._symmetry)

    @classmethod
    def from_dense(cls, ham, *, symmetry=None):
        symmetry = ham.symmetry if symmetry is None else symmetry
        g = _chemist_symmetric(ham)
        if symmetry is not None and symmetry != ham.symmetry:
            _validate_dense_blocks(g, check_symmetry(symmetry, ham.nbasis))
        p, q = np.tril_indices(ham.nbasis)
        return cls._from_pairs(ham.ecore, ham.one_mo, p, q,
                               lambda i, j: g[p[i], q[i], p[j], q[j]], symmetry, validated=True)

    @classmethod
    def from_pair_matrix(cls, ecore, one, matrix, *, symmetry=None, symmetry_atol=_ATOL):
        """Pack a fourfold chemist pair matrix with checked forbidden elements."""
        matrix = np.asarray(matrix)
        if np.iscomplexobj(matrix):
            raise TypeError('CSF integrals must be real')
        p, q = np.tril_indices(len(one))
        if matrix.shape != (len(p), len(p)):
            raise ValueError('invalid fourfold orbital-pair matrix dimensions')
        if not np.isfinite(symmetry_atol) or symmetry_atol < 0.:
            raise ValueError('symmetry tolerance must be finite and nonnegative')
        if symmetry is not None:
            ecore, one = _one_electron(ecore, one, symmetry, symmetry_atol)
        return cls._from_pairs(ecore, one, p, q,
                               lambda i, j: .5*matrix[i, j] + .5*matrix[j, i], symmetry,
                               raw=lambda i, j: matrix[i, j], symmetry_atol=symmetry_atol)

    @classmethod
    def _from_pairs(cls, ecore, one, p, q, row, symmetry, raw=None, symmetry_atol=_ATOL,
                    validated=False):
        check_symmetry(symmetry, len(one))
        pair_ids = np.arange(len(p))
        blocks = (pair_ids,)
        if cls._block_packed:
            if symmetry is None:
                raise TypeError('block-packed ERIs require a PointGroup')
            blocks = symmetry._pair_blocks()[2]
        elif raw is not None and symmetry is not None:
            for ids in symmetry._pair_blocks()[2]:
                if len(ids):
                    other = np.setdiff1d(pair_ids, ids, assume_unique=True)
                    for i in ids:
                        _check_forbidden(raw(i, other), symmetry_atol)
            # The global constructor validates its payload at the default
            # tolerance. Canonicalize accepted residuals before passing it in.
            codes = np.asarray(symmetry.orbital_irreps, dtype=np.intp)
            codes = codes[p] ^ codes[q]
            unmasked = row
            row = lambda i, j: np.where(codes[i] == codes[j], unmasked(i, j), 0.)
        packed = np.empty(sum(len(ids)*(len(ids)+1)//2 for ids in blocks))
        begin = 0
        for ids in blocks:
            other = np.setdiff1d(pair_ids, ids, assume_unique=True) if len(ids) and not validated else ()
            for j, i in enumerate(ids):
                if len(other):
                    _check_forbidden(row(i, other) if raw is None else raw(i, other), symmetry_atol)
                packed[begin:begin+j+1] = row(i, ids[:j+1])
                begin += j+1
        return cls(ecore, one, packed, symmetry=symmetry)


class BlockPackedHamiltonian(PackedHamiltonian):
    """Eightfold triangles of the allowed pair-irrep blocks, concatenated by code."""

    _block_packed = True

    def __init__(self, ecore, one, packed, *, symmetry):
        if symmetry is None:
            raise TypeError('block-packed ERIs require a PointGroup')
        super().__init__(ecore, one, packed, symmetry=symmetry)

    @classmethod
    def from_packed(cls, ham, *, symmetry=None):
        """Pack an existing eightfold triangle into its point-group blocks."""
        if not isinstance(ham, PackedHamiltonian):
            raise TypeError('expected packed Hamiltonian integrals')
        symmetry = ham.symmetry if symmetry is None else symmetry
        if isinstance(ham, BlockPackedHamiltonian):
            if symmetry == ham.symmetry:
                return cls(ham.ecore, ham.one_mo, ham.packed_eri, symmetry=symmetry)
            ham = ham.to_global_packed(symmetry=False)
        p, q = np.tril_indices(ham.nbasis)

        def row(i, j):
            high = np.maximum(i, j)
            return ham.packed_eri[high*(high+1)//2 + np.minimum(i, j)]

        return cls._from_pairs(ham.ecore, ham.one_mo, p, q, row, symmetry,
                               validated=symmetry == ham.symmetry)

    def to_global_packed(self, *, symmetry=True):
        pairs = self.nbasis*(self.nbasis+1)//2
        packed = np.zeros(pairs*(pairs+1)//2)
        for code, ids in enumerate(self.symmetry._pair_blocks()[2]):
            begin = self.block_offsets[code]
            for count, i in enumerate(ids, start=1):
                packed[i*(i+1)//2 + ids[:count]] = self.packed_eri[begin:begin+count]
                begin += count
        return PackedHamiltonian(self.ecore, self.one_mo, packed,
                                 symmetry=self.symmetry if symmetry else None)


def save_hamiltonian(filename, ham, *, nelec=None):
    """Save active integrals and optional electron count to a NumPy snapshot."""
    if not isinstance(ham, (Hamiltonian, PackedHamiltonian)):
        raise TypeError('expected an SCSF Hamiltonian')
    fields = dict(ecore=ham.ecore, one=ham.one_mo)
    if isinstance(ham, BlockPackedHamiltonian):
        fields['block_packed_eri'] = ham.packed_eri
    elif isinstance(ham, PackedHamiltonian):
        fields['packed_eri'] = ham.packed_eri
    else:
        fields['two'] = ham.two_mo
    if nelec is not None:
        fields['nelec'] = nelec
    if ham.symmetry is not None:
        fields.update(symmetry_version=1, point_group=ham.symmetry.group,
                      orbital_irreps=np.asarray(ham.symmetry.orbital_irreps, dtype=np.intp))
    np.savez(filename, **fields)


def load_hamiltonian(filename):
    """Read active integrals and the optional electron count from a snapshot."""
    with np.load(filename, allow_pickle=False) as data:
        fields = {'symmetry_version', 'point_group', 'orbital_irreps'} & set(data.files)
        symmetry = None
        if fields:
            if len(fields) != 3 or data['symmetry_version'].item() != 1:
                raise ValueError('unsupported or incomplete snapshot symmetry metadata')
            symmetry = PointGroup(data['point_group'].item(), data['orbital_irreps'])
        nelec = int(data['nelec']) if 'nelec' in data else None
        formats = {'two': Hamiltonian, 'packed_eri': PackedHamiltonian,
                   'block_packed_eri': BlockPackedHamiltonian}
        payloads = formats.keys() & set(data.files)
        if len(payloads) != 1:
            raise ValueError('snapshot needs exactly one of two, packed_eri or block_packed_eri')
        key, = payloads
        return formats[key](data['ecore'].item(), data['one'], data[key], symmetry=symmetry), nelec


def rotate_hamiltonian(ham, rotation, *, orbital_irreps=None, preserve_symmetry=True):
    """Transform the active orbitals by a square orthogonal matrix of column vectors."""
    rotation = np.asarray(rotation)
    symmetry = ham.symmetry if preserve_symmetry else None
    if symmetry is not None:
        if (np.iscomplexobj(rotation) or rotation.shape != ham.one_mo.shape or
                not np.allclose(rotation.T @ rotation, np.eye(ham.nbasis), atol=1e-10, rtol=0.)):
            raise ValueError('expected a real orthogonal orbital rotation')
        old_labels = np.asarray(symmetry.orbital_irreps, dtype=np.intp)
        if orbital_irreps is None:
            orbital_irreps = old_labels[np.argmax(np.abs(rotation), axis=0)] if len(old_labels) else ()
        symmetry = type(symmetry)(symmetry.group, orbital_irreps)._check_size(ham.nbasis)
        new_labels = np.asarray(symmetry.orbital_irreps, dtype=np.intp)
        forbidden = old_labels[:, None] != new_labels
        if np.any(np.abs(rotation[forbidden]) > _ATOL):
            raise ValueError('rotation mixes orbital irreps')
        rotation = np.where(forbidden, 0., rotation)
    one = rotation.T @ ham.one_mo @ rotation
    one = (one + one.T) * .5
    if isinstance(ham, BlockPackedHamiltonian):
        if symmetry is None:
            return rotate_hamiltonian(ham.to_global_packed(symmetry=False), rotation)
        return _rotate_blocks(ham, rotation, symmetry, one)
    if isinstance(ham, PackedHamiltonian) and symmetry is not None:
        return _rotate_blocks(BlockPackedHamiltonian.from_packed(ham), rotation,
                              symmetry, one).to_global_packed()
    if isinstance(ham, Hamiltonian):
        two = np.einsum('pi,rj,qk,sl,prqs->ijkl', rotation, rotation, rotation, rotation,
                        ham.two_mo, optimize=True)
        two = (two + two.swapaxes(0, 2)) * .5
        two = (two + two.swapaxes(1, 3)) * .5
        two = (two + two.transpose(1, 0, 3, 2)) * .5
        return Hamiltonian(ham.ecore, one, two, symmetry=symmetry)
    p, q = np.tril_indices(ham.nbasis)
    pair_index = np.empty((ham.nbasis, ham.nbasis), dtype=int)
    pair_index[p, q] = pair_index[q, p] = np.arange(len(p))

    def rotate_pair(values):
        matrix = rotation.T @ values[pair_index] @ rotation
        return (matrix[p, q] + matrix[q, p]) * .5

    if isinstance(ham, PackedHamiltonian):
        pairs = np.arange(len(p))
        offsets = pairs * (pairs + 1) // 2
        half = np.empty((len(p), len(p)), order='F')
        # Rotate one pair at a time in each index of the ERI pair matrix.
        for i in pairs:
            indices = offsets[np.maximum(i, pairs)] + np.minimum(i, pairs)
            half[:, i] = rotate_pair(ham.packed_eri[indices])
        packed = np.empty_like(ham.packed_eri)
        for i, row in enumerate(half):
            packed[offsets[i]:offsets[i] + i + 1] = rotate_pair(row)[:i + 1]
        return PackedHamiltonian(ham.ecore, one, packed, symmetry=symmetry)
    raise TypeError('expected dense or packed Hamiltonian integrals')


def _rotate_blocks(ham, rotation, symmetry, one):
    _, _, old_blocks = ham.symmetry._pair_blocks()
    _, _, new_blocks = symmetry._pair_blocks()
    packed = np.empty_like(ham.packed_eri)
    old_codes = np.asarray(ham.symmetry.orbital_irreps, dtype=np.intp)
    new_codes = np.asarray(symmetry.orbital_irreps, dtype=np.intp)
    extent = len(symmetry.names)
    old_orbitals = [np.flatnonzero(old_codes == g) for g in range(extent)]
    new_orbitals = [np.flatnonzero(new_codes == g) for g in range(extent)]
    if list(map(len, old_orbitals)) != list(map(len, new_orbitals)):
        raise ValueError('rotation changes the number of orbitals in an irrep')
    rotations = [rotation[np.ix_(a, b)] for a, b in zip(old_orbitals, new_orbitals)]

    def pair_ids(a, b):
        high = np.maximum(a[:, None], b)
        return high*(high+1)//2 + np.minimum(a[:, None], b)

    # The two passes transform each allowed orbital-irrep rectangle.
    # Keep one fourfold intermediate block, never a global pair matrix.
    for code, (old, new) in enumerate(zip(old_blocks, new_blocks)):
        size = len(old)
        if not size:
            continue
        begin = ham.block_offsets[code]
        labels = np.arange(size)
        offsets = labels*(labels+1)//2
        rectangles = []
        for g, left in enumerate(old_orbitals):
            h = g ^ code
            if h < g or not len(left) or not len(old_orbitals[h]):
                continue
            a = np.searchsorted(old, pair_ids(left, old_orbitals[h]))
            b = np.searchsorted(new, pair_ids(new_orbitals[g], new_orbitals[h]))
            rectangles.append((a, b, rotations[g], rotations[h], g == h))

        def transform(values):
            result = np.empty(size)
            for a, b, left, right, symmetric in rectangles:
                rotated = left.T @ values[a] @ right
                result[b] = .5*(rotated + rotated.T) if symmetric else rotated
            return result

        half = np.empty((size, size), order='F')
        for j in labels:
            indices = offsets[np.maximum(j, labels)] + np.minimum(j, labels)
            half[:, j] = transform(ham.packed_eri[begin + indices])
        for j, row in enumerate(half):
            packed[begin+offsets[j]:begin+offsets[j]+j+1] = transform(row)[:j+1]
    return BlockPackedHamiltonian(ham.ecore, one, packed, symmetry=symmetry)
