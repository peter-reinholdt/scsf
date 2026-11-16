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



hamiltonian = Hamiltonian
