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
python -m scsf --xyz examples/water.xyz --basis sto-3g --point-group --target-irrep A1
python -m scsf --integrals scsf/test/data/lih_sto3g.npz --natorb sci
python -m scsf --xyz examples/water.xyz --basis sto-3g --ncore 1 --natorb ccsd
python -m scsf --integrals scsf/test/data/lih_sto3g.npz --kernel cih --selection-unit configuration
python -m scsf.drivers.polarizability --xyz examples/water.xyz --basis sto-3g
```

The installed commands are `scsf` and `scsf-polarizability`. N−2 is the
default Hamiltonian kernel. HCI selects individual spin couplings by default.
Use `--selection-unit configuration` to add complete occupations.
The energy driver stops after adding a batch of at most 1% of the current CSF
count and converging all requested roots in the enlarged space. Set
`--stop-growth-fraction 0` for strict selection through a zero-addition step.
The driver right-pads `--reference` with zeros to fill the active orbital space.

Use `--point-group` to enable real orbital irreps for XYZ input. Supply a
subgroup such as `--point-group C2v` to choose it explicitly. The supported
groups are `D2h` and its subgroups. Use `--target-irrep A1` (or the subgroup's
numeric irrep code) to restrict the CI. The default seed is chosen in that
sector, and an explicit reference must belong to it. A labeled calculation
without a target retains seed-driven selection. The polarizability driver
uses the same options and grows response spaces in the sectors connected by the property.

Use `--natorb sci` (or bare `--natorb`) to generate natural orbitals from a
pilot SCI run with the same selection settings. The driver diagonalizes the
spin-summed 1-RDM, averaged equally over the requested roots, and starts a new
SCI run in descending natural occupation order. SCI natural orbitals work with
XYZ and integral snapshots. With XYZ input, `--natorb mp2` and `--natorb ccsd`
use the corresponding PySCF 1-RDM before running SCI. These modes honor
`--ncore`. All modes retain every active orbital and interpret `--reference`
in the orbital order of each run. With point-group symmetry, the natural
orbital rotation diagonalizes the 1-RDM within each orbital irrep.

Both commands default to one native worker per available physical core,
up to 256. Set `--nthreads` to override the worker count, which includes
the calling thread. On Linux, `--worker-cpus` pins each energy driver worker
to a distinct allowed logical CPU in the order listed. For example,
`--nthreads 2 --worker-cpus 4 6` pins workers to CPUs 4 and 6 if both are
in the process's allowed CPU set.

NumPy and SciPy use their linked BLAS library for Davidson linear algebra.
Set `OPENBLAS_NUM_THREADS` for pthreads OpenBLAS, `MKL_NUM_THREADS` for MKL,
or `OMP_NUM_THREADS` for an OpenMP BLAS before launching the driver.
`--nthreads` controls the native SCSF workers independently.

`--max-mem 4GiB` sets one shared limit for the tracked native allocation
sizes, including HCI transfer tables, schedules, spin coefficients, caches,
and the full RHS block's product arrays. The default is unlimited. An
insufficient limit raises a memory error. Process RSS also includes the
integral input, the basis, Python and Davidson arrays, and other workspace.
The Python constructor accepts the same limit as `max_mem` in bytes.

The energy driver prints HCI growth, per-iteration Davidson timings and a
final phase table by default. Use `--quiet` for summaries and the table, or
`--sci-stats` for detailed counters. The energy driver also supports an exact
leading-block preconditioner and Jacobi–Davidson corrections.

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

Use `op.native_rdm1(c)` and `op.native_rdm2(c)` for spin-summed density
matrices. For transition densities, use `op.native_transition_rdm1(bra, ket)`
and `op.native_transition_rdm2(bra, ket)`. The routines take real coefficients
in the same native CSF basis. Convert public-order vectors with
`op.to_native_order(c)`. The 1-RDM is indexed by creation and annihilation
orbital `[p,q]`. The 2-RDM uses physicist order `[p,r,q,s]` and contracts
with dense ERIs as `0.5 * np.sum(ham.two_mo * rdm2)`. Both 2-RDM routines
accept `max_output_bytes`, `max_link_bytes` and `max_workspace_bytes`.
The default output cap is 256 MiB. The combined `op.native_rdms(c)` also returns memory counters.
`layout="blocks"` returns a `BlockRDM2` with directed orbital-pair blocks.
Use `to_dense()` to restore physicist order or `eri_projection()` to sum the
density entries that share each eightfold ERI. Pass `block_packed=False`
to get the projection in global eightfold order. Use
`op.native_density_energy(c)` or `op.native_density_energy(bra, ket)` to
contract directly without allocating the four-index density. Pass
`include_ecore=True` to include the core energy times the bra–ket overlap.

For fixed-space calculations, construct the operator with a saved basis
from `csf_wfn.from_file()` or enumerate the spin sector with `add_all_csfs()`.

`csf_one_electron_product` applies spin-scalar spatial operators, including
complex nonsymmetric matrices. The energy operator consumes dense, global
eightfold-packed, or point-group block-packed integrals. Dense ERIs use
`two[p,r,q,s] = (pq|rs)`.

Construct orbital metadata with `scsf.PointGroup("C2v", orbital_irreps)`.
`scsf.BlockPackedHamiltonian.from_dense(ham, symmetry=group)` packs the
lower triangle of each allowed pair-irrep block. Supply `symmetry=group`
and optionally `target_irrep` when constructing a `csf_wfn` or querying
`csf_dimension`. Array imports check symmetry-forbidden elements against
an absolute tolerance of `1e-12`. The PySCF molecular driver allows
`1e-10` hartree for residuals from numerical geometry and integration.

Use `scsf.save_hamiltonian` and `scsf.load_hamiltonian` for integral snapshots.
Snapshots store `ecore`, `one`, optional `nelec`, and one ERI payload
(`two`, `packed_eri`, or `block_packed_eri`). Labeled snapshots also
store the subgroup and active orbital irreps.

[Water growth and fixed-space timings](docs/scsf-v2-million-solves-2026-11-20.md).

## Development

Run `python -m pytest -q` after building. Tests cover signed Hamiltonian and
property products, selection, response and the command-line drivers.

SCSF is licensed under GPL-3.0-or-later. Bundled pybind11 has its own license
in `LICENSE.pybind11`.
