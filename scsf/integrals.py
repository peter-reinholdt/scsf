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


class _StructuredIntegrals:
    def __init__(self, ecore, one):
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


def _chemist_symmetric(ham, atol=1e-12):
    g = ham.two_mo.transpose(0, 2, 1, 3)
    if not all(np.allclose(g, other, rtol=0., atol=atol)
               for other in (g.swapaxes(0, 1), g.swapaxes(2, 3), g.transpose(2, 3, 0, 1))):
        raise ValueError('ERI tensor must have real spin-free permutation symmetry')
    return g


class CholeskyHamiltonian(_StructuredIntegrals):
    """Real MO integrals represented by Coulomb-metric Cholesky or DF factors.

    Supply factors of shape (rank, nbasis, nbasis) or (rank, nbasis*(nbasis+1)//2)
    in NumPy lower-triangle order. No sqrt(2) weighting of off-diagonal pairs is
    used: (pq|rs) is the dot product of the factors for pq and rs.
    """

    def __init__(self, ecore, one, factors):
        super().__init__(ecore, one)
        if np.iscomplexobj(factors):
            raise TypeError('CSF integrals must be real')
        k = self.nbasis
        factors = np.asarray(factors, dtype=float)
        if factors.ndim == 3 and factors.shape[1:] == (k, k):
            if not np.allclose(factors, factors.swapaxes(1, 2), rtol=0., atol=1e-12):
                raise ValueError('Cholesky factor matrices must be symmetric')
            p, q = np.tril_indices(k)
            factors = factors[:, p, q]
        if factors.ndim != 2 or factors.shape[1] != k*(k+1)//2:
            raise ValueError('Cholesky factors must have shape (rank, nbasis, nbasis) or (rank, orbital pairs)')
        if not np.all(np.isfinite(factors)):
            raise ValueError('integrals must be finite')
        self.pair_factors = np.array(factors.T, dtype=float, order='C', copy=True)
        self.pair_factors.flags.writeable = False
        self.max_eri_error = None

    def _structured_eri(self):
        return self.pair_factors, None

    @classmethod
    def from_dense(cls, ham, tolerance=1e-10):
        """Pivoted Cholesky of an existing dense Hamiltonian's Coulomb matrix.

        The absolute residual diagonal cutoff is in Hartree. The measured
        maximum absolute residual over packed ERIs is stored as max_eri_error.
        This conversion uses a dense pair matrix during setup.
        """
        from scipy.linalg.lapack import dpstrf

        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.:
            raise ValueError('Cholesky tolerance must be finite and positive')
        g = _chemist_symmetric(ham, min(tolerance, 1e-12))
        p, q = np.tril_indices(ham.nbasis)
        eri = g[p[:, None], q[:, None], p, q]
        n = len(p)
        factors = np.empty((n, 0))
        if np.max(np.diag(eri), initial=0.) > tolerance:
            lower, pivots, rank, _ = dpstrf(eri, tol=tolerance, lower=1)
            factors = np.empty((n, rank))
            factors[pivots-1] = np.tril(lower[:, :rank])
        error = np.max(np.abs(eri-factors @ factors.T), initial=0.)
        roundoff = np.finfo(float).eps * max(1, n) * np.max(np.abs(eri), initial=0.)
        if not np.isfinite(error) or error > tolerance + roundoff:
            raise ValueError('ERI matrix cannot be Cholesky factorized to the requested tolerance')
        result = cls(ham.ecore, ham.one_mo, factors.T)
        result.max_eri_error = float(error)
        return result


class PackedHamiltonian(_StructuredIntegrals):
    """Exact eightfold-packed real ERIs, with both triangles in NumPy order."""

    def __init__(self, ecore, one, packed):
        super().__init__(ecore, one)
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

    def _structured_eri(self):
        return self.packed_eri, None


class THCHamiltonian(_StructuredIntegrals):
    """Real tensor hypercontraction, (pq|rs) = Σ_uv X[p,u] X[q,u] Z[u,v] X[r,v] X[s,v]."""

    def __init__(self, ecore, one, collocation, core):
        super().__init__(ecore, one)
        if np.iscomplexobj(collocation) or np.iscomplexobj(core):
            raise TypeError('CSF integrals must be real')
        self.collocation = np.array(collocation, dtype=float, order='C', copy=True)
        self.core = np.array(core, dtype=float, order='C', copy=True)
        if self.collocation.ndim != 2 or self.collocation.shape[0] != self.nbasis:
            raise ValueError('THC collocation must have shape (nbasis, rank)')
        rank = self.collocation.shape[1]
        if self.core.shape != (rank, rank):
            raise ValueError('THC core must have shape (rank, rank)')
        if not all(np.all(np.isfinite(v)) for v in (self.collocation, self.core)):
            raise ValueError('integrals must be finite')
        if not np.allclose(self.core, self.core.T, rtol=0., atol=1e-12):
            raise ValueError('THC core must be symmetric')
        self.collocation.flags.writeable = self.core.flags.writeable = False
        self.max_eri_error = None

    @classmethod
    def fit(cls, ham, orbital_values, rank, *, rcond=1e-12):
        """Fit a THC core to dense ERIs with QR-selected orbital collocation columns.

        orbital_values[p,u] is the value of active MO p at candidate point u.
        Pair rows are weighted by their multiplicities in the full tensor.
        The measured maximum packed ERI residual is stored as max_eri_error.
        """
        from operator import index
        from scipy.linalg import qr

        if np.iscomplexobj(orbital_values):
            raise TypeError('orbital values must be real')
        x = np.asarray(orbital_values, dtype=float)
        rank = index(rank)
        if x.ndim != 2 or x.shape[0] != ham.nbasis or not np.all(np.isfinite(x)):
            raise ValueError('orbital values must be finite with shape (nbasis, points)')
        if not np.isfinite(rcond) or not 0 < rcond < 1:
            raise ValueError('rcond must lie strictly between zero and one')
        p, q = np.tril_indices(ham.nbasis)
        if not 1 <= rank <= min(x.shape[1], len(p)):
            raise ValueError('THC rank must lie between one and the number of points or orbital pairs')
        g = _chemist_symmetric(ham)[p[:, None], q[:, None], p, q]
        weight = np.where(p == q, 1., np.sqrt(2.))
        norms = np.linalg.norm(x, axis=0)
        x = x/np.where(norms > 0., norms, 1.)
        pair_values = x[p]*x[q]
        _, _, pivots = qr(weight[:, None]*pair_values, mode='economic', pivoting=True)
        selected = pivots[:rank]
        selected_pairs = pair_values[:, selected]
        fit = np.linalg.pinv(weight[:, None]*selected_pairs, rcond=rcond)*weight
        core = fit @ g @ fit.T
        result = cls(ham.ecore, ham.one_mo, x[:, selected], (core+core.T)*.5)
        result.max_eri_error = float(np.max(np.abs(g-selected_pairs @ result.core @ selected_pairs.T)))
        return result

    def _structured_eri(self):
        return self.collocation, self.core


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
    if isinstance(ham, THCHamiltonian):
        return THCHamiltonian(ham.ecore, one, rotation.T @ ham.collocation, ham.core)

    p, q = np.tril_indices(ham.nbasis)
    pair_index = np.empty((ham.nbasis, ham.nbasis), dtype=int)
    pair_index[p, q] = pair_index[q, p] = np.arange(len(p))

    def rotate_pair(values):
        matrix = rotation.T @ values[pair_index] @ rotation
        return (matrix[p, q] + matrix[q, p]) * .5

    if isinstance(ham, CholeskyHamiltonian):
        rotated = np.empty(ham.pair_factors.T.shape)
        for result, factors in zip(rotated, ham.pair_factors.T):
            result[:] = rotate_pair(factors)
        return CholeskyHamiltonian(ham.ecore, one, rotated)
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
    raise TypeError('expected dense, packed, Cholesky or THC integrals')
