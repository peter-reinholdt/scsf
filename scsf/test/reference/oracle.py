"""Independent explicit determinant actions for small signed acceptance checks."""
from functools import lru_cache
from itertools import combinations, product
from math import sqrt

import numpy as np
import scsf
from scsf.csf import _configuration


def random_hamiltonian(k=4, seed=17):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(k, k))
    h = (h + h.T) / 2
    g = rng.normal(size=(k,) * 4)
    g = (g + g.swapaxes(0, 1)) / 2
    g = (g + g.swapaxes(2, 3)) / 2
    g = (g + g.transpose(2, 3, 0, 1)) / 2
    return scsf.hamiltonian(1.2345, h, g.transpose(0, 2, 1, 3))

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

def fock_action(occupied, annihilate, create):
    """Independent list-based fermionic action, with creation-order signs."""
    occupied = list(occupied)
    sign = 1
    for p in annihilate:
        if p not in occupied:
            return (None, 0)
        sign *= (-1) ** occupied.index(p)
        occupied.remove(p)
    for p in create:
        if p in occupied:
            return (None, 0)
        sign *= (-1) ** sum((q < p for q in occupied))
        occupied.append(p)
        occupied.sort()
    return (tuple(occupied), sign)

@lru_cache(maxsize=4)
def fock_reference(ham, nelec, two_ms):
    """Build H and S² by explicit second-quantized actions (no SC rules)."""
    k = ham.nbasis
    na, nb = ((nelec + two_ms) // 2, (nelec - two_ms) // 2)
    dets = [a + tuple((p + k for p in b)) for a in combinations(range(k), na) for b in combinations(range(k), nb)]
    lookup = {d: i for i, d in enumerate(dets)}
    h = np.zeros((len(dets), len(dets)))
    s2 = np.eye(len(dets)) * (two_ms / 2) * (two_ms / 2 + 1)
    for j, det in enumerate(dets):
        for p, q, sigma in product(range(k), range(k), range(2)):
            target, phase = fock_action(det, (q + sigma * k,), (p + sigma * k,))
            if phase:
                h[lookup[target], j] += phase * ham.one_mo[p, q]
        terms = product(range(k), range(k), range(k), range(k), range(2), range(2))
        for p, r, q, s, sigma, tau in terms:
            target, phase = fock_action(det, (q + sigma * k, s + tau * k), (r + tau * k, p + sigma * k))
            if phase:
                h[lookup[target], j] += 0.5 * phase * ham.two_mo[p, r, q, s]
        for q, p in product(range(k), repeat=2):
            middle, sign1 = fock_action(det, (q + k,), (q,))
            if sign1:
                target, sign2 = fock_action(middle, (p,), (p + k,))
                if sign2:
                    s2[lookup[target], j] += sign1 * sign2
    keys = [sum((1 << p for p in det)) for det in dets]
    return (keys, h, s2)

def _reference_columns(columns, keys):
    lookup = {key: row for row, key in enumerate(keys)}
    result = np.zeros((len(keys), len(columns)))
    for col, amplitudes in enumerate(columns):
        for key, value in amplitudes.items():
            result[lookup[key], col] = value
    return result

def transform_in_order(wfn, keys, two_ms):
    return _reference_columns([_expand(label, two_ms) for label in wfn], keys)

def hamiltonian_block(ham, rows, cols, two_s):
    rows, cols = tuple(rows), tuple(cols)
    if not rows or not cols:
        return np.zeros((len(rows), len(cols)))
    wfn = scsf.csf_wfn(ham.nbasis, sum(_configuration(rows[0])), two_s)
    for label in (*rows, *cols):
        wfn.add_csf(label)
    keys, matrix, _ = fock_reference(ham, wfn.nelec, two_s)
    transform = transform_in_order(wfn, keys, two_s)
    left = [wfn.index_csf(label) for label in rows]
    right = [wfn.index_csf(label) for label in cols]
    return transform[:, left].T @ matrix @ transform[:, right]


def fock_operators(wfn):
    """Independent explicit spin-orbital actions in the existing CG convention."""
    columns = [_expand(label, wfn.two_s) for label in wfn]
    keys = sorted({key for column in columns for key in column})
    u = _reference_columns(columns, keys)
    dets = [tuple(p for p in range(2*wfn.nbasis) if key & (1 << p))
            for key in keys]
    lookup = {det: i for i, det in enumerate(dets)}

    def matrix(indices):
        k = wfn.nbasis
        result = np.zeros((len(dets), len(dets)))
        for j, det in enumerate(dets):
            if len(indices) == 2:
                p, q = indices
                actions = [((q+spin*k,), (p+spin*k,)) for spin in range(2)]
            else:
                p, q, r, s = indices
                actions = [((q+sigma*k, s+tau*k), (r+tau*k, p+sigma*k))
                           for sigma, tau in product(range(2), repeat=2)]
            for annihilate, create in actions:
                target, sign = fock_action(det, annihilate, create)
                if sign and target in lookup:
                    result[lookup[target], j] += sign
        return u.T @ result @ u

    return matrix
