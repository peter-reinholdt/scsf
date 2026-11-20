#!/usr/bin/env python

import numpy as np

def davidson_response(A, b, hdiag, tol=1e-4, maxiter=100, verbose=False, guess=None):
    """Galerkin Davidson solve for real or damped complex response."""
    b = np.asarray(b)
    if np.allclose(b, 0., atol=1e-20):
        return b

    def matvec(v):
        result = A(v.real)
        if np.any(v.imag):
            result = result + 1j*A(v.imag)
        return result

    # The shifted diagonal can have zero entries, including in the projected
    # direction. Preserve the phase wherever a denominator is nonzero.
    diagonal = np.asarray(hdiag, dtype=np.complex128).copy()
    magnitude = np.abs(diagonal)
    small = (magnitude>0.) & (magnitude<1e-12)
    diagonal[small] *= 1e-12/magnitude[small]
    diagonal[magnitude==0.] = 1e-12
    seed = b/diagonal if guess is None else np.asarray(guess, dtype=np.complex128)
    if not np.any(seed):
        seed = b/diagonal
    V = (seed/np.linalg.norm(seed))[:, None]
    AV = matvec(V[:, 0])[:, None]
    for i in range(maxiter):
        dual = V.conj().T
        reduced = dual@AV
        rhs = dual@b
        try:
            coefficients = np.linalg.solve(reduced, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(reduced, rhs, rcond=None)[0]
        residual = b-AV@coefficients
        norm = np.linalg.norm(residual)
        if verbose:
            print(f'Iteration {i+1:3d} residual norm: {norm:6e}')
        if norm<tol:
            return V@coefficients
        if i+1==maxiter or V.shape[1]==len(b):
            break
        correction = residual/diagonal
        for _ in range(2):
            correction -= V@(dual@correction)
        correction_norm = np.linalg.norm(correction)
        if correction_norm<=1e-12*norm:
            correction = residual-V@(dual@residual)
            correction_norm = np.linalg.norm(correction)
        if correction_norm<=1e-15*norm:
            break
        correction /= correction_norm
        V = np.column_stack((V, correction))
        AV = np.column_stack((AV, matvec(correction)))
    raise ValueError('response Davidson did not converge')


def solve_ci(hvp, hdiag, roots, tol=1e-6, maxiter=100, verbose=False, c0=None,
             reference_state=None, max_subspace=None, stats=None,
             preconditioner_block=None, jd_iterations=0):
    """Block Davidson for a real symmetric electronic Hamiltonian.

    Optional ``stats`` is filled with nonoverlapping wall timings, product
    widths, iteration and restart counts. Product validation is included in
    product time; remaining bookkeeping is reported as ``other_ns``.

    ``hvp`` accepts a matrix of trial vectors as columns. Returns electronic
    eigenvalues, column-oriented eigenvectors, and a convergence flag. ``c0``
    may be a vector or a block, including zero-padded guesses after selection.
    An optional exact principal block replaces the diagonal preconditioner
    on its selected rows.
    Positive ``jd_iterations`` uses bounded MINRES corrections in the
    orthogonal complement of each Ritz vector. The outer residual still
    uses the supplied exact operator.
    No core-energy constant is added here. ``reference_state`` retains roots
    through the first state with absolute overlap above 0.9, as in the original
    routine; convergence requires finding that state when tracking is requested.

    ``hvp`` must be linear. Independently validate residuals when using a
    screened direct operator.
    """
    from operator import index
    from time import perf_counter_ns
    started = perf_counter_ns()
    if stats is not None:
        stats.clear()
        stats.update(product_ns=0, orthogonalization_ns=0, ritz_ns=0, restart_ns=0,
                     product_calls=0, product_columns=0, jd_product_calls=0, iterations=0, restarts=0, call_widths=[])
    def timed(name):
        def decorate(function):
            if stats is None:
                return function
            def measured(*args, **kwargs):
                before = perf_counter_ns()
                try:
                    return function(*args, **kwargs)
                finally:
                    stats[name] += perf_counter_ns() - before
            return measured
        return decorate
    def finish(values, vectors, converged):
        if stats is not None:
            stats['total_ns'] = perf_counter_ns() - started
            stats['other_ns'] = stats['total_ns'] - sum(stats[k] for k in
                ('product_ns', 'orthogonalization_ns', 'ritz_ns', 'restart_ns'))
            stats['converged'] = converged
        return values, vectors, converged

    hdiag = np.asarray(hdiag)
    if np.iscomplexobj(hdiag) or hdiag.ndim != 1 or not np.all(np.isfinite(hdiag)):
        raise ValueError('hdiag must be a finite real vector')
    hdiag = np.asarray(hdiag, dtype=float)
    dim, roots = len(hdiag), index(roots)
    if not 1 <= roots <= dim or not np.isfinite(tol) or tol <= 0 or maxiter < 1:
        raise ValueError('require 1 <= roots <= dimension, tol > 0 and maxiter >= 1')
    maxiter = index(maxiter)
    jd_iterations = index(jd_iterations)
    if jd_iterations<0:
        raise ValueError('jd_iterations must be nonnegative')
    if jd_iterations:
        from scipy.sparse.linalg import LinearOperator, minres
    if preconditioner_block is not None:
        block_rows, block = preconditioner_block
        block_rows = np.asarray(block_rows, dtype=np.intp)
        block = np.asarray(block)
        if (block_rows.ndim != 1 or block.shape != (block_rows.size,)*2 or
                not block_rows.size or np.any(block_rows < 0) or np.any(block_rows >= dim) or
                np.unique(block_rows).size != block_rows.size or not np.isrealobj(block) or
                not np.isfinite(block).all() or not np.allclose(block, block.T, atol=1e-12, rtol=0)):
            raise ValueError('preconditioner needs distinct row indices and a finite symmetric block')
        block_values, block_vectors = np.linalg.eigh((block+block.T)*.5)

    def shifted_denominator(values, shifts):
        denominators = shifts[None, :]-values[:, None]
        small = np.abs(denominators)<1e-3
        denominators[small] = np.where(denominators[small]<0., -1e-3, 1e-3)
        return denominators

    def inverse(vector, den, local_den):
        out = vector/den
        if preconditioner_block is not None:
            out[block_rows] = block_vectors@((block_vectors.T@vector[block_rows])/local_den)
        return out

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

    @timed('orthogonalization_ns')
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

    @timed('product_ns')
    def apply(vectors):
        if stats is not None:
            stats['product_calls'] += 1
            stats['product_columns'] += vectors.shape[1]
            stats['call_widths'].append(vectors.shape[1])
        result = np.asarray(hvp(vectors))
        if (result.shape != vectors.shape or np.iscomplexobj(result) or
                not np.all(np.isfinite(result))):
            raise ValueError('hvp must return a finite real block with the input shape')
        return result

    def jd_correction(v, shift, residual):
        def project(x):
            return x-v*np.dot(v, x)

        def action(x):
            z = project(x)
            if stats is not None:
                stats['jd_product_calls'] += 1
            return project(apply(z[:, None])[:, 0]-shift*z) + (x-z)

        inverse_diag = 1./np.maximum(np.abs(hdiag-shift), 1e-3)

        def precondition(x):
            z = project(x)
            return project(inverse_diag*z) + (x-z)

        operator = LinearOperator((dim, dim), matvec=action, dtype=float)
        preconditioner = LinearOperator((dim, dim), matvec=precondition, dtype=float)
        rhs = -project(residual)
        correction, info = minres(operator, rhs, M=preconditioner, maxiter=jd_iterations)
        if info>=0 and np.isfinite(correction).all():
            return project(correction)
        return None

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
    for iteration in range(maxiter):
        ritz_start = perf_counter_ns() if stats is not None else 0
        if stats is not None: stats['iterations'] = iteration + 1
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
        if stats is not None: stats['ritz_ns'] += perf_counter_ns() - ritz_start
        if verbose:
            for k in range(len(L)):
                print(f'{iteration+1}       {k+1}   {L[k]:.12e}    {norms[k]:.6e}')
        if np.all(norms < tol) and found_reference:
            return finish(L, X, True)
        if iteration+1 == maxiter:
            return finish(L, X, False)
        unconverged = np.flatnonzero(norms >= tol)
        shifts = L[unconverged]
        denom = shifted_denominator(hdiag, shifts)
        local_den = None if preconditioner_block is None else shifted_denominator(block_values, shifts)
        corrections = inverse(residual[:, unconverged], denom, local_den)
        if jd_iterations:
            for column, root in enumerate(unconverged):
                corrected = jd_correction(X[:, root], L[root], residual[:, root])
                if corrected is not None and np.any(corrected):
                    corrections[:, column] = corrected
        corrections = orthogonalize(corrections, V)
        if not corrections.shape[1] and V.shape[1] < dim:
            corrections = supplement(V, max(1, roots))
        if not corrections.shape[1]:
            return finish(L, X, False)
        if V.shape[1]+corrections.shape[1] > max_subspace:
            # Thick restart retains the current Ritz vectors, without applying
            # H a second time to the retained columns.
            if X.shape[1] >= max_subspace:
                return finish(L, X, False)
            if stats is not None:
                stats['restarts'] += 1
                restart_start, orthogonal_before = perf_counter_ns(), stats['orthogonalization_ns']
            V, AV = X, AX
            corrections = orthogonalize(corrections, V)
            if stats is not None:
                stats['restart_ns'] += perf_counter_ns() - restart_start - (stats['orthogonalization_ns'] - orthogonal_before)
        corrections = corrections[:, :max_subspace-V.shape[1]]
        if not corrections.shape[1]:
            return finish(L, X, False)
        correction_images = apply(corrections)
        V = np.column_stack((V, corrections))
        AV = np.column_stack((AV, correction_images))
    return finish(L, X, False)
