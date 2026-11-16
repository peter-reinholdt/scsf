# SCSF

Spin-adapted selected configuration interaction using genealogical configuration
state functions. The C++ backend provides CG/GUGA couplings, selection, stored
CSR Hamiltonians and direct matrix-vector products. Python drivers provide
Davidson CI/SCI and variational singlet polarizabilities.

The Python API is loosely based on and inspired by PyCI.

## Build

Python 3.10+, a C++17 compiler and NumPy/SciPy are required. Pybind11 headers are
bundled. The native extension uses standard C++ threads.

```sh
python -m pip install --no-build-isolation -e .
# Or build an offline checkout with its existing Python dependencies:
python setup.py build_ext --inplace
```

Install `.[molecule]` for XYZ/PySCF inputs and `.[profile]` for profiling tools.
`EXTRA_CXXFLAGS` can supply compiler options such as `-march=native`.

## Run

```sh
python -m scsf --integrals scsf/test/data/lih_sto3g.npz --full
python -m scsf --xyz examples/water.xyz --basis cc-pvdz --ncore 1 --direct
python -m scsf --help
python -m scsf.drivers.polarizability --xyz examples/water.xyz --basis sto-3g
```

The installed commands are `scsf` and `scsf-polarizability`. Stored CSR is the
default; `--direct` selects compact-spin GUGA. Both drivers start fresh.

For example, apply the direct Hamiltonian to a full CSF space:

```python
import numpy as np
import scsf

with np.load("scsf/test/data/lih_sto3g.npz", allow_pickle=False) as data:
    ham = scsf.hamiltonian(float(data["ecore"]), data["one"], data["two"])
    nelec = int(data["nelec"])
wfn = scsf.csf_wfn(ham.nbasis, nelec=nelec, two_s=0)
wfn.add_all_csfs()
op = scsf.csf_direct_op(ham, wfn)
y = op.matvec(np.ones(len(wfn)))
```

Starting with the same `ham` and `nelec`, a singlet HCI calculation grows a
selected space from the closed-shell reference. Each iteration solves for the
lowest state, selects CSFs with `|H_ai c_i| > eps`, and updates the operator:

```python
from scsf.drivers.solvers import solve_ci

wfn = scsf.csf_wfn(ham.nbasis, nelec=nelec, two_s=0)
wfn.add_csf("2" * (nelec // 2) + "0" * (ham.nbasis - nelec // 2))
op = scsf.csf_op(ham, wfn)  # Use csf_direct_op here for direct products.
guess = None
for iteration in range(50):
    energies, coeffs, converged = solve_ci(
        op.matmat, op.diagonal(), roots=1, c0=guess, tol=1e-10, maxiter=200)
    if not converged:
        raise RuntimeError("Davidson did not converge")
    energy = energies[0] + ham.ecore  # Add the constant energy once.
    print(f"iteration={iteration} ncsf={len(wfn)} energy={energy:.12f}")
    added = scsf.add_csf_hci(ham, wfn, coeffs[:, 0], eps=1e-4, backend=op.backend)
    if added == 0:
        break
    op.update(ham, wfn)
    guess = np.pad(coeffs, ((0, added), (0, 0)))
else:
    raise RuntimeError("HCI selection did not converge")
```

This LiH/STO-3G example reaches -7.882324378883 Hartree with 41 selected CSFs
(out of 105 singlet CSFs). Selection stops when no more CSFs pass the threshold.
[HCI example](examples/hci.py) accepts `--eps`, `--direct` and an integral snapshot:

```sh
python -m examples.hci
python -m examples.hci --direct
```

Integral arrays use `two[p,r,q,s] = (pq|rs)`. Driver `.npz` snapshots contain
`ecore`, `one`, `two` and optionally `nelec`. 

## Development and measurements

Run `python -m pytest -q` after building. Tests are in `scsf/test`.

SCSF is licensed under GPL-3.0-or-later. Bundled pybind11 has its own license
in `LICENSE.pybind11`.
