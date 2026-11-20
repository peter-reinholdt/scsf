"""A small HCI calculation using the core SCSF Python routines.

From a built source checkout, run ``python -m examples.hci`` for LiH/STO-3G,
or pass an integral snapshot and optionally select a kernel.
"""
import argparse
from pathlib import Path

import numpy as np
import scsf
from scsf.drivers.response import grow_space
from scsf.drivers.solvers import solve_ci


def hci(ham, nelec, *, eps=1e-4, kernel='nminus2', max_iterations=50):
    """Return the selected singlet basis, total energy and coefficient vector."""
    if nelec % 2:
        raise ValueError('this singlet example requires an even electron count')
    wfn = scsf.csf_wfn(ham.nbasis, nelec=nelec, two_s=0)
    wfn.add_csf('2' * (nelec // 2) + '0' * (ham.nbasis - nelec // 2))
    op = scsf.csf_structured_op(ham, wfn, strategy=kernel)
    guess = None
    for iteration in range(max_iterations):
        energies, coeffs, converged = solve_ci(
            op.matmat, op.diagonal(), roots=1, c0=guess, tol=1e-10, maxiter=200)
        if not converged:
            raise RuntimeError('Davidson did not converge')
        energy = energies[0] + ham.ecore
        print(f'iteration={iteration} ncsf={len(wfn)} energy={energy:.12f} hartree')
        added = grow_space(wfn, op, np.abs(coeffs[:, 0]), eps)
        if added == 0:
            return wfn, energy, coeffs[:, 0]
        guess = np.pad(coeffs, ((0, added), (0, 0)))
    raise RuntimeError('HCI selection did not converge within the iteration limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('integrals', nargs='?', type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        'scsf/test/data/lih_sto3g.npz')
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--kernel', choices=scsf.csf_structured_op.kernels, default='nminus2')
    args = parser.parse_args()
    with np.load(args.integrals, allow_pickle=False) as data:
        ham = scsf.hamiltonian(float(data['ecore']), data['one'], data['two'])
        nelec = int(data['nelec'])
    hci(ham, nelec, eps=args.eps, kernel=args.kernel)


if __name__ == '__main__':
    main()
