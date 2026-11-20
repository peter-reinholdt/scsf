# SCSF; licensed under GPL-3.0-or-later.
import scsf
import numpy as np
from .solvers import davidson_response, solve_ci
from collections import defaultdict

def one_electron_ao2mo(casci, operator, mo_coeff=None, ncas=None, ncore=None):
    if mo_coeff is None:
        mo_coeff = casci.mo_coeff
    if ncas is None:
        ncas = casci.ncas
    if ncore is None:
        ncore = casci.ncore
    mo_cas = mo_coeff[:, ncore:ncore + ncas]
    return mo_cas.conj().T @ operator @ mo_cas

def wrap_matvec(op, x):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        result = wrap_matvec(op, x.real).astype(complex)
        if np.any(x.imag):
            result += 1j * wrap_matvec(op, x.imag)
        return result
    if x.ndim == 1:
        return op.matvec(np.ascontiguousarray(x))
    if x.ndim == 2:
        return op.matmat(x)
    raise ValueError('products require a vector or column block')

def zeropad(x, N):
    shape = (N,) + x.shape[1:]
    out = np.zeros(shape, dtype=x.dtype)
    out[:len(x), ...] = x
    return out

def grow_space(wfn, op, weights, eps):
    """Apply native HCI while keeping the response vectors in public order."""
    old_size = len(wfn)
    _, stats = op.select_hci(op.to_native_order(weights), eps)
    if stats['added']:
        op.append_selected_labels(wfn, old_size)
    return stats['added']


def resp(ham, wfn, op, e_vecs, integrals, perturbations, gamma=0.0, triplet=False, eps_mu=None, eps_resp=None, overwrite=True, nroots=1, state=0, reference_state=None, *, ci_tol=1e-10, response_tol=1e-08, response_maxiter=200, response_growth_tol=0.01, verbose=True):
    """Variational linear response in fixed-spin CSF spaces.

    CSF properties are singlet/spin-scalar one-electron operators. Property
    selection uses immutable parent coefficients; Hamiltonian response growth
    uses native HCI. No PT2 contribution is included.
    Return the selected space/operator, CI columns, response/property vectors,
    and response functions.
    """
    from copy import copy
    if not isinstance(wfn, scsf.csf_wfn) or not isinstance(op, scsf.csf_structured_op):
        raise TypeError('response requires a CSF space and its structured Hamiltonian')
    if len(wfn) != op.shape[0] or any(a != b for a, b in zip(wfn, op.labels())):
        raise ValueError('response space must match the Hamiltonian row order')
    if triplet:
        raise NotImplementedError('triplet one-electron CSF response is not implemented')
    if not len(wfn) or state < 0 or nroots < 1 or (not np.isfinite(gamma)) or (gamma < 0) or (not np.isfinite(response_growth_tol)) or (not 0 <= response_growth_tol < 1) or (not np.isfinite(ci_tol)) or (ci_tol <= 0) or (not np.isfinite(response_tol)) or (response_tol <= 0) or (response_maxiter < 1):
        raise ValueError('invalid response state, damping or growth tolerance')
    perturbations = tuple(dict.fromkeys(perturbations))
    for label, omega, parity in perturbations:
        if label not in integrals or not np.isfinite(omega) or parity not in (-1, 0, 1):
            raise ValueError('invalid perturbation label, frequency or parity')
    integrals = {label: np.asarray(one) for label, one in integrals.items()}
    for one in integrals.values():
        if one.shape != (ham.nbasis, ham.nbasis) or not np.all(np.isfinite(one)):
            raise ValueError('property integrals must be finite spatial orbital matrices')
    if not overwrite:
        wfn = copy(wfn)
        op = op.with_basis(wfn)
    if np.iscomplexobj(e_vecs):
        raise ValueError('CI coefficients must be real for the real Hamiltonian solver')
    e_vecs = np.array(e_vecs, dtype=float, copy=True)
    if e_vecs.ndim == 1:
        e_vecs = e_vecs[:, None]
    if e_vecs.ndim != 2 or e_vecs.shape[0] != len(wfn):
        raise ValueError('CI coefficients must have one row per original basis state')
    nroots = max(nroots, state + 1)
    nthreads = op.nthreads

    def diagonalize():
        nonlocal e_vecs, state
        energies, e_vecs, ok = solve_ci(lambda x: wrap_matvec(op, x), op.diagonal(), roots=nroots, c0=e_vecs, tol=ci_tol, maxiter=500, reference_state=reference_state)
        if not ok:
            raise RuntimeError('CI Davidson did not converge in the response space')
        if reference_state is not None:
            overlaps = np.abs(np.asarray(reference_state).conj() @ e_vecs[:len(reference_state)])
            matched = np.flatnonzero(overlaps > 0.9)
            if len(matched):
                state = int(matched[0])
        return energies

    def property_product(label, vector):
        return scsf.csf_one_electron_product(integrals[label], wfn, vector, nthreads=nthreads)
    energies = diagonalize()
    if eps_mu is not None and integrals:
        original = e_vecs[:, state].copy()
        added = 0
        for label, one in integrals.items():
            added += scsf.add_csf_one_electron(one, wfn, zeropad(original, len(wfn)), eps=eps_mu, nthreads=nthreads)
        e_vecs = zeropad(e_vecs, len(wfn))
        op.update(ham, wfn)
        energies = diagonalize()
        if verbose:
            print(f'couple_property added={added} basis={len(wfn)} energy={energies[state] + op.ecore:.12f}')
    property_vectors, response_vectors = {}, {}

    def solve_responses(previous=None):
        nonlocal property_vectors, response_vectors
        c = e_vecs[:, state]
        e0 = energies[state]

        def project(x):
            return x - c * np.vdot(c, x)
        property_vectors = {label: project(property_product(label, c)) for label in dict.fromkeys((p[0] for p in perturbations))}
        response_vectors = {}
        ci_diagonal = op.diagonal()
        for key in perturbations:
            label, omega, parity = key
            shift = parity * (omega + 1j * gamma)

            def action(x):
                qx = project(x)
                return project(wrap_matvec(op, qx) - (e0 + shift) * qx) + c * np.vdot(c, x)
            diagonal = ci_diagonal - e0 - shift + np.abs(c)**2
            guess = None if previous is None else project(zeropad(previous[key], len(wfn)))
            x = davidson_response(action, property_vectors[label], diagonal, tol=response_tol, maxiter=response_maxiter, guess=guess)
            x = project(x)
            residual = action(x) - property_vectors[label]
            if np.linalg.norm(residual) > response_tol * 1.01:
                raise RuntimeError('projected response residual exceeds tolerance')
            response_vectors[key] = x
    solve_responses()
    if eps_resp is not None and response_vectors:
        while True:
            weights = np.max(np.abs(np.array(list(response_vectors.values()))), axis=0)
            added = grow_space(wfn, op, weights, eps_resp)
            if not added:
                break
            e_vecs = zeropad(e_vecs, len(wfn))
            energies = diagonalize()
            previous = response_vectors
            solve_responses(previous)
            if verbose:
                print(f'couple_response added={added} basis={len(wfn)} energy={energies[state] + op.ecore:.12f}')
            if added / len(wfn) < response_growth_tol:
                break
    response_functions = defaultdict(float)
    for label, omega, parity in perturbations:
        factor = 2 if omega == 0 and gamma == 0 and (parity == 0) else 1
        for label2, b in property_vectors.items():
            response_functions[label, label2, omega] += factor * np.vdot(b, response_vectors[label, omega, parity])
    return (wfn, op, e_vecs, response_vectors, property_vectors, response_functions)

def print_var_summary(response_functions):
    for (label1, label2, omega), value in response_functions.items():
        name = f'<<{label1}; {label2}>>(ω={omega})'
        if value.imag:
            print(f'Re{name} value={value.real:16.9f}')
            print(f'Im{name} value={value.imag:16.9f}')
        else:
            print(f'{name} value={value.real:16.9f}')
