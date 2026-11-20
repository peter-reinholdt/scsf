# SCSF

Spin-adapted selected configuration interaction using genealogically coupled
configuration state functions. The native occupation-block operator implements
exact N−2, CIH, N−1 and particle–hole Hamiltonian products. Python drivers
provide Davidson CI, HCI selection, and variational singlet
polarizabilities.

## Build

Python 3.10+, a C++17 compiler and NumPy/SciPy are required. Pybind11 headers are
bundled. The native extension uses standard C++ threads.

```sh
python -m pip install --no-build-isolation -e .
# For an offline checkout with its Python dependencies already installed:
python setup.py build_ext --inplace
```

Install `.[molecule]` for XYZ/PySCF inputs and `.[profile]` for benchmark tools.
`EXTRA_CXXFLAGS` can supply compiler options such as `-march=native`.

## Energy and response

```sh
python -m scsf --integrals scsf/test/data/lih_sto3g.npz
python -m scsf --xyz examples/water.xyz --basis cc-pvdz --ncore 1 --reference 2222
python -m scsf --integrals scsf/test/data/lih_sto3g.npz --kernel cih --selection-unit configuration
python -m scsf.drivers.polarizability --xyz examples/water.xyz --basis sto-3g
```

The installed commands are `scsf` and `scsf-polarizability`. N−2 is the
default Hamiltonian kernel. HCI selects individual spin couplings by default.
Use `--selection-unit configuration` to add complete occupations.
The driver right-pads `--reference` with zeros to fill the active orbital space.

Both commands default to one native worker per available physical core,
up to 256. Set `--nthreads` to override the worker count, which includes
the calling thread. On Linux, `--worker-cpus` pins each energy driver worker
to a distinct allowed logical CPU in the order listed. For example,
`--nthreads 2 --worker-cpus 4 6` pins workers to CPUs 4 and 6 if both are
in the process's allowed CPU set.

`--max-plan-bytes` option caps cached CIH plans or particle–hole neighbors.
Use `--sci-stats` to print product counters, Davidson residuals and HCI
selection statistics. The energy driver also supports an exact leading-block
preconditioner and Jacobi–Davidson corrections.

## Python API

```python
import numpy as np
import scsf

with np.load("scsf/test/data/lih_sto3g.npz", allow_pickle=False) as data:
    ham = scsf.Hamiltonian(float(data["ecore"]), data["one"], data["two"])
    nelec = int(data["nelec"])

wfn = scsf.csf_wfn(ham.nbasis, nelec, 0)
wfn.add_all_csfs()
op = scsf.csf_structured_op(ham, wfn, strategy="nminus2")
y = op.matvec(np.ones(len(wfn)))
```

`matvec`, `matmat` and `diagonal` use the submitted CSF order. The
`native_*` methods and the HCI selector use occupation-block order.
The driver remaps the previous Davidson vectors when selected rows move.
[HCI example](examples/hci.py).

For fixed-space calculations, construct the operator with a saved basis
from `csf_wfn.from_file()` or enumerate the spin sector with `add_all_csfs()`.

`csf_one_electron_product` applies spin-scalar spatial operators, including
complex nonsymmetric matrices. The energy operator consumes dense, packed,
Cholesky and THC integrals directly. Dense ERIs use `two[p,r,q,s] = (pq|rs)`.
Driver snapshots contain `ecore`, `one` and optionally `nelec), together with
one of `two`, `packed_eri`, `cholesky_factors` (rank first) or
`thc_collocation` and `thc_core`.

[Water growth and fixed-space timings](docs/scsf-v2-million-solves-2026-11-20.md).

## Development

Run `python -m pytest -q` after building. Tests cover signed Hamiltonian and
property products, selection, response and the command-line drivers.

SCSF is licensed under GPL-3.0-or-later. Bundled pybind11 has its own license
in `LICENSE.pybind11`.
