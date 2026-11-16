# Independent molecular references

## LiH/STO-3G

`scsf/test/data/lih_sto3g.npz` is generated from Li at (0, 0, 0) and H at
(0, 0, 1.6) Angstrom, charge zero, using all six canonical RHF orbitals and
all four electrons. No integral data are imported from another project.

Regenerate it with NumPy and PySCF installed:

```sh
python scsf/test/reference/generate_lih.py
```

The generator uses PySCF alone for the integrals and three FCI roots in each
spin sector, checking every reference state's spin. The stored references
are used by tests without requiring PySCF at test time. With PySCF 2.11.0,
the total energies in Hartree are:

| Root | Singlet | Triplet |
| --- | --- | --- |
| 0 | -7.88232437888350 | -7.76666900957178 |
| 1 | -7.74941469373004 | -7.71658823812218 |
| 2 | -7.69719315103717 | -7.71658823812216 |

RHF gives -7.86186476980865 Hartree. The fixture also exercises HCI selection
and incremental Hamiltonian updates, native and direct products, and
conversion of CSF coefficients to determinants.

## Dalton H4/STO-3G

The `.dal` and `.mol` inputs here produce singlet and triplet spin-adapted CI
roots for a linear H4/STO-3G system. The NumPy fixture is
`scsf/test/data/h4_dalton_sto3g.npz`. Dalton's constant energy is written to eight
decimal places; regression tests compare absolute roots at that precision and
spin gaps more tightly. These inputs and reference data were produced
independently for the CSF validation.

```sh
python -m scsf --integrals scsf/test/data/h4_dalton_sto3g.npz --full --nroots 3
python -m scsf --integrals scsf/test/data/h4_dalton_sto3g.npz --full --two-s 2 --nroots 3
```
