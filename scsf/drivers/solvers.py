#!/usr/bin/env python

import numpy as np

def davidson_response(A, b, hdiag, tol=1e-4, maxiter=100, verbose=False, guess=None):
    if np.allclose(b, 0.0, atol=1e-20):
        return b

    def matvec(v, A=A):
        if not np.any(v.imag):
            Av = A(v.real)
        else:
            Av_real = A(v.real)
            Av_imag = A(v.imag)
            Av = Av_real + 1j*Av_imag
        return Av
    if guess is None:
        diagonal_guess = b/(hdiag)
        diagonal_guess /= np.linalg.norm(diagonal_guess)
        guess = diagonal_guess
    V = guess.astype(np.complex128).reshape(-1, 1)
    AV = np.zeros_like(V, dtype=np.complex128)
    AV[:, 0] = matvec(V[:, 0])

    bred = V.T @ b
    for i in range(maxiter):
        Ered = V.T @ AV
        xred = np.linalg.solve(Ered, bred)
        x = xred @ V.T
        residual = b - (xred @ AV.T)
        if verbose:
            print(f'Iteration {i+1:3d} residual norm: {np.linalg.norm(residual):6e}')
        if np.linalg.norm(residual) < tol:
            return x
        delta = residual/(hdiag)
        delta = delta/np.linalg.norm(delta)
        vnew = delta - V @ (V.T @ delta)
        vnew = vnew/np.linalg.norm(vnew)
        V = np.hstack([V, vnew[:, None]])
        bred = V.T @ b
        AV = np.hstack([AV, matvec(vnew)[:, None]])
    raise ValueError('Not converged')

def solve_ci(hvp, hdiag, roots, tol=1e-6, maxiter=100, verbose=False, c0=None,
             reference_state=None, max_subspace=None, trial_vector_tol=0.0,
             trial_vector_decay=0.0, trial_vector_source='orthogonal'):
    """Block Davidson for a real symmetric electronic Hamiltonian.

    ``hvp`` accepts a matrix of trial vectors as columns. Returns electronic
    eigenvalues, column-oriented eigenvectors, and a convergence flag. ``c0``
    may be a vector or a block, including zero-padded guesses after selection.
    No core-energy constant is added here. ``reference_state`` retains roots
    through the first state with absolute overlap above 0.9, as in the original
    routine; convergence requires finding that state when tracking is requested.

    ``trial_vector_tol`` optionally zeros components smaller than this absolute
    threshold in each *normalized orthogonal correction*, before reprojecting
    onto the Davidson basis. At least the largest component is kept. Products
    are taken on these sparse vectors; the corresponding orthonormal directions
    and their images are recovered by the same linear transformations using
    cached basis images. Thus clamping changes the search directions, not the
    Hamiltonian or convergence criterion. Dependent sparse directions fall back
    to ordinary corrections. Guesses and returned Ritz vectors are not clamped.
    Zero (the default) retains the original algorithm. ``hvp`` must be linear;
    independently validate residuals when using a screened direct operator.
    A positive ``trial_vector_decay`` tapers the cutoff by
    ``min(1, residual_norm / initial_max_residual)**trial_vector_decay`` for
    each root, restoring fuller corrections as convergence improves. Zero
    keeps the cutoff fixed. The reference scale is reset at each solve.
    ``trial_vector_source='preconditioned'`` instead clamps before the initial
    orthogonalization; ``'orthogonal'`` (default) clamps the proposed new trial
    direction. Both recover consistent images through cached projections.
    """
    from operator import index

    hdiag = np.asarray(hdiag)
    if np.iscomplexobj(hdiag) or hdiag.ndim != 1 or not np.all(np.isfinite(hdiag)):
        raise ValueError('hdiag must be a finite real vector')
    hdiag = np.asarray(hdiag, dtype=float)
    dim, roots = len(hdiag), index(roots)
    if not 1 <= roots <= dim or not np.isfinite(tol) or tol <= 0 or maxiter < 1:
        raise ValueError('require 1 <= roots <= dimension, tol > 0 and maxiter >= 1')
    maxiter = index(maxiter)
    if not np.isfinite(trial_vector_tol) or not 0 <= trial_vector_tol < 1:
        raise ValueError('trial_vector_tol must be finite and in [0, 1)')
    if not np.isfinite(trial_vector_decay) or trial_vector_decay < 0:
        raise ValueError('trial_vector_decay must be finite and nonnegative')
    if trial_vector_source not in ('orthogonal', 'preconditioned'):
        raise ValueError("trial_vector_source must be 'orthogonal' or 'preconditioned'")
    if max_subspace is None:
        max_subspace = min(dim, max(20, 8*roots))
    max_subspace = index(max_subspace)
    if not roots <= max_subspace <= dim or (roots < dim and max_subspace == roots):
        raise ValueError('max_subspace must allow roots and at least one correction')
    if reference_state is not None:
        reference_state = np.asarray(reference_state)
        if (np.iscomplexobj(reference_state) or reference_state.ndim != 1 or
                not 0 < len(reference_state) <= dim or
                not np.all(np.isfinite(reference_state)) or
                np.linalg.norm(reference_state) == 0):
            raise ValueError('reference_state must be a finite nonzero real prefix vector')
        reference_state = reference_state / np.linalg.norm(reference_state)

    def orthogonalize(vectors, against):
        accepted = []
        for vector in vectors.T:
            vector = vector.copy()
            scale = np.linalg.norm(vector)
            if scale == 0:
                continue
            vector /= scale
            # Two passes matter after zero padding and near convergence.
            for _ in range(2):
                if against.shape[1]:
                    vector -= against @ (against.T @ vector)
                for q in accepted:
                    vector -= q*np.dot(q, vector)
            norm = np.linalg.norm(vector)
            if norm > 1e-12:
                accepted.append(vector/norm)
        return np.column_stack(accepted) if accepted else np.empty((dim, 0))

    def apply(vectors):
        result = np.asarray(hvp(vectors))
        if (result.shape != vectors.shape or np.iscomplexobj(result) or
                not np.all(np.isfinite(result))):
            raise ValueError('hvp must return a finite real block with the input shape')
        return result

    def orthogonalize_transform(vectors, against, min_norm=1e-12):
        # Return D = vectors @ T + against @ P, using the same two-pass
        # orthogonalization on all three factors. Carrying the transformation
        # lets us form H D from sparse H inputs and cached H against, without
        # either clamping an orthonormal vector or treating H(clamp(D)) as H D.
        accepted, transforms, projections = [], [], []
        for j, source in enumerate(vectors.T):
            vector = source.copy()
            scale = np.linalg.norm(vector)
            if scale == 0:
                continue
            vector /= scale
            transform = np.zeros(vectors.shape[1])
            transform[j] = 1/scale
            projection = np.zeros(against.shape[1])
            for _ in range(2):
                if against.shape[1]:
                    coefficients = against.T @ vector
                    vector -= against @ coefficients
                    projection -= coefficients
                for q, t, p in zip(accepted, transforms, projections):
                    coefficient = np.dot(q, vector)
                    vector -= coefficient*q
                    transform -= coefficient*t
                    projection -= coefficient*p
            norm = np.linalg.norm(vector)
            if norm > min_norm:
                accepted.append(vector/norm)
                transforms.append(transform/norm)
                projections.append(projection/norm)
        if not accepted:
            return np.empty((dim, 0)), None, None
        return (np.column_stack(accepted), np.column_stack(transforms),
                np.column_stack(projections))

    def sparse_corrections(vectors, basis, images, cutoffs):
        sparse = vectors.copy()
        for column, cutoff in zip(sparse.T, cutoffs):
            norm = np.linalg.norm(column)
            if norm:
                column /= norm
                largest = int(np.argmax(np.abs(column)))
                keep = column[largest]
                column[np.abs(column) < cutoff] = 0.
                column[largest] = keep
                column /= np.linalg.norm(column)
        # Avoid amplifying product roundoff in nearly dependent sparse inputs.
        directions, transform, projection = orthogonalize_transform(sparse, basis, min_norm=1e-8)
        if not directions.shape[1]:
            return directions, None
        used = np.any(transform != 0, axis=1)
        product = apply(sparse[:, used])
        return directions, product @ transform[used] + images @ projection

    empty = np.empty((dim, 0))
    if c0 is None:
        V = empty
    else:
        c0 = np.asarray(c0)
        if c0.ndim == 1:
            c0 = c0[:, None]
        if (np.iscomplexobj(c0) or c0.ndim != 2 or c0.shape[0] != dim or
                not np.all(np.isfinite(c0))):
            raise ValueError('c0 must be a finite real vector/block of the full dimension')
        V = orthogonalize(np.asarray(c0, dtype=float), empty)[:, :max_subspace]
    diagonal_order = np.argsort(hdiag, kind='stable')

    def supplement(basis, count):
        # Coordinate vectors also recover directions when a diagonal Davidson
        # preconditioner collapses onto the current span.
        extra = []
        for p in diagonal_order:
            unit = np.zeros((dim, 1))
            unit[p, 0] = 1.0
            current = np.column_stack([basis] + extra) if extra else basis
            q = orthogonalize(unit, current)
            if q.shape[1]:
                extra.append(q)
            if len(extra) >= count:
                break
        return np.column_stack(extra) if extra else empty

    # Include low-diagonal directions even with supplied guesses. A zero-padded
    # guess can otherwise remain an exact higher state of a disconnected block.
    nseed = min(roots, max_subspace-V.shape[1])
    if nseed:
        V = np.column_stack((V, supplement(V, nseed)))
    AV = apply(V)
    initial_residual = None
    for iteration in range(maxiter):
        projected = V.T @ AV
        values, rotations = np.linalg.eigh((projected+projected.T)/2)
        nkeep, found_reference = roots, reference_state is None
        if reference_state is not None:
            overlaps = np.abs(reference_state @ (V[:len(reference_state)] @ rotations))
            matches = np.flatnonzero(overlaps > 0.9)
            if len(matches):
                nkeep, found_reference = max(roots, int(matches[0])+1), True
        L, Z = values[:nkeep], rotations[:, :nkeep]
        X, AX = V @ Z, AV @ Z
        residual = AX-X*L[None, :]
        norms = np.linalg.norm(residual, axis=0)
        if initial_residual is None:
            initial_residual = max(float(np.max(norms)), np.finfo(float).tiny)
        if verbose:
            for k in range(len(L)):
                print(f'{iteration+1}       {k+1}   {L[k]:.12e}    {norms[k]:.6e}')
        if np.all(norms < tol) and found_reference:
            return L, X, True
        if iteration+1 == maxiter:
            return L, X, False
        unconverged = np.flatnonzero(norms >= tol)
        denom = L[unconverged][None, :]-hdiag[:, None]
        small = np.abs(denom) < 1e-3
        denom[small] = np.where(denom[small] < 0, -1e-3, 1e-3)
        corrections = residual[:, unconverged]/denom
        correction_images = None
        if trial_vector_tol:
            raw_corrections = corrections
            cutoffs = trial_vector_tol*np.minimum(1., norms[unconverged]/initial_residual)**trial_vector_decay
            if trial_vector_source == 'orthogonal':
                # Project first to prevent large components already represented
                # in V from consuming most of a sparse direction's support.
                # Handle columns separately so per-root cutoffs stay aligned
                # even if one direction is rank deficient.
                trial = []
                for column in corrections.T:
                    q = orthogonalize(column[:, None], V)
                    trial.append(q[:, 0] if q.shape[1] else np.zeros(dim))
                corrections = np.column_stack(trial) if trial else empty
            corrections, correction_images = sparse_corrections(corrections, V, AV, cutoffs)
            if not corrections.shape[1]:
                corrections = orthogonalize(raw_corrections, V)
        else:
            corrections = orthogonalize(corrections, V)
        if not corrections.shape[1] and V.shape[1] < dim:
            corrections = supplement(V, max(1, roots))
        if not corrections.shape[1]:
            return L, X, False
        if V.shape[1]+corrections.shape[1] > max_subspace:
            # Thick restart retains the current Ritz vectors, without applying
            # H a second time to the retained columns.
            if X.shape[1] >= max_subspace:
                return L, X, False
            V, AV = X, AX
            if correction_images is None:
                corrections = orthogonalize(corrections, V)
            else:
                corrections, transform, projection = orthogonalize_transform(corrections, V)
                if corrections.shape[1]:
                    correction_images = correction_images @ transform + AV @ projection
        corrections = corrections[:, :max_subspace-V.shape[1]]
        if not corrections.shape[1]:
            return L, X, False
        if correction_images is None:
            correction_images = apply(corrections)
        else:
            correction_images = correction_images[:, :corrections.shape[1]]
        V = np.column_stack((V, corrections))
        AV = np.column_stack((AV, correction_images))
    return L, X, False
