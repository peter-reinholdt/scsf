"""Real spatial integrals in the CSF convention two[p,r,q,s] = (pq|rs)."""
import numpy as np


class Hamiltonian:
    """Own real integral arrays in physicist order.

    Construct with ``Hamiltonian(ecore, one, two)`` where
    ``two[p,r,q,s] = (pq|rs)``. Arrays are copied into contiguous owned storage.
    """
    def __init__(self, ecore, one, two):
        if one is None or two is None:
            raise TypeError('provide ecore, one and two arrays')
        if any(np.iscomplexobj(value) for value in (ecore, one, two)):
            raise TypeError('CSF integrals must be real')
        self.ecore = float(ecore)
        self.one_mo = np.array(one, dtype=float, order='C', copy=True)
        self.two_mo = np.array(two, dtype=float, order='C', copy=True)
        if self.one_mo.ndim != 2 or self.one_mo.shape[0] != self.one_mo.shape[1]:
            raise ValueError('one-electron integrals must be a square matrix')
        self.nbasis = len(self.one_mo)
        if self.two_mo.shape != (self.nbasis,)*4:
            raise ValueError('two-electron integrals must have shape (nbasis,)*4')
        if not all(np.all(np.isfinite(x)) for x in (self.ecore, self.one_mo, self.two_mo)):
            raise ValueError('integrals must be finite')
        if not np.allclose(self.one_mo, self.one_mo.T, rtol=0., atol=1e-12):
            raise ValueError('one-electron integrals must be symmetric')
        _chemist_symmetric(self)
        self.one_mo.flags.writeable = self.two_mo.flags.writeable = False


hamiltonian = Hamiltonian


def _chemist_symmetric(ham, atol=1e-12):
    g = ham.two_mo.transpose(0, 2, 1, 3)
    if not all(np.allclose(g, other, rtol=0., atol=atol)
               for other in (g.swapaxes(0, 1), g.swapaxes(2, 3), g.transpose(2, 3, 0, 1))):
        raise ValueError('ERI tensor must have real spin-free permutation symmetry')
    return g


class PackedHamiltonian:
    """Exact eightfold-packed real ERIs, with both triangles in NumPy order."""

    def __init__(self, ecore, one, packed):
        if np.iscomplexobj(ecore) or np.iscomplexobj(one):
            raise TypeError('CSF integrals must be real')
        self.ecore = float(ecore)
        self.one_mo = np.array(one, dtype=float, order='C', copy=True)
        if self.one_mo.ndim != 2 or self.one_mo.shape[0] != self.one_mo.shape[1]:
            raise ValueError('one-electron integrals must be a square matrix')
        if not np.isfinite(self.ecore) or not np.all(np.isfinite(self.one_mo)):
            raise ValueError('integrals must be finite')
        if not np.allclose(self.one_mo, self.one_mo.T, rtol=0., atol=1e-12):
            raise ValueError('one-electron integrals must be symmetric')
        self.nbasis = len(self.one_mo)
        self.one_mo.flags.writeable = False
        if np.iscomplexobj(packed):
            raise TypeError('CSF integrals must be real')
        pairs = self.nbasis*(self.nbasis+1)//2
        self.packed_eri = np.array(packed, dtype=float, order='C', copy=True)
        if self.packed_eri.shape != (pairs*(pairs+1)//2,):
            raise ValueError('packed ERIs must be the lower triangle of the orbital-pair matrix')
        if not np.all(np.isfinite(self.packed_eri)):
            raise ValueError('integrals must be finite')
        self.packed_eri.flags.writeable = False

    @classmethod
    def from_dense(cls, ham):
        g = _chemist_symmetric(ham)
        p, q = np.tril_indices(ham.nbasis)
        packed = np.empty(len(p)*(len(p)+1)//2)
        for i, (a, b) in enumerate(zip(p, q)):
            begin = i*(i+1)//2
            packed[begin:begin+i+1] = g[a,b,p[:i+1],q[:i+1]]
        return cls(ham.ecore, ham.one_mo, packed)


def rotate_hamiltonian(ham, rotation):
    """Transform the active orbitals by a square orthogonal matrix of column vectors."""
    one = rotation.T @ ham.one_mo @ rotation
    one = (one + one.T) * .5
    if isinstance(ham, Hamiltonian):
        two = np.einsum('pi,rj,qk,sl,prqs->ijkl', rotation, rotation, rotation, rotation,
                        ham.two_mo, optimize=True)
        two = (two + two.swapaxes(0, 2)) * .5
        two = (two + two.swapaxes(1, 3)) * .5
        two = (two + two.transpose(1, 0, 3, 2)) * .5
        return Hamiltonian(ham.ecore, one, two)
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
        return PackedHamiltonian(ham.ecore, one, packed)
    raise TypeError('expected dense or packed Hamiltonian integrals')
