"""Point-group blocks of the directed spin-free two-particle density."""
import numpy as np


class BlockRDM2:
    """A 2-RDM with blocks indexed by the irreps of directed pairs (p,q) and (r,s).

    Entries have the same value as dense storage at [p,r,q,s]. Pair lists
    retain both orientations, including for real transition densities.
    """

    def __init__(self, nbasis, symmetry, values, transfer_irreps):
        self.nbasis = nbasis
        self.symmetry = symmetry
        self.transfer_irreps = tuple(code for code in range(8) if transfer_irreps & (1 << code))
        labels = np.zeros(nbasis, dtype=np.intp) if symmetry is None else np.asarray(
            symmetry.orbital_irreps, dtype=np.intp)
        codes = (labels[:, None] ^ labels).ravel()
        extent = 1 if symmetry is None else len(symmetry.names)
        self.pairs = tuple(np.column_stack(np.divmod(np.flatnonzero(codes == code), nbasis))
                           for code in range(extent)) if nbasis else tuple(
                               np.empty((0, 2), dtype=np.intp) for _ in range(extent))
        for pairs in self.pairs:
            pairs.flags.writeable = False
        self.blocks = {}
        begin = 0
        for g, left in enumerate(self.pairs):
            for h, right in enumerate(self.pairs):
                if (g ^ h) in self.transfer_irreps and len(left) and len(right):
                    end = begin + len(left)*len(right)
                    self.blocks[g, h] = values[begin:end].reshape(len(left), len(right))
                    begin = end
        if begin != values.size:
            raise ValueError('density block sizes do not match the data')

    def to_dense(self):
        """Return a dense 2-RDM in physicist index order."""
        out = np.zeros((self.nbasis,)*4)
        for (g, h), values in self.blocks.items():
            p, q = self.pairs[g].T
            r, s = self.pairs[h].T
            out[p[:, None], r, q[:, None], s] = values
        return out

    def eri_projection(self, *, block_packed=True):
        """Sum distinct density entries at their canonical ERI addresses.

        For scalar ERIs in the matching representation, the two-electron
        contraction is half the dot product with this projection.
        """
        k = self.nbasis
        pairs = k*(k+1)//2
        use_blocks = block_packed and self.symmetry is not None
        blocks = self.symmetry._pair_blocks()[2] if use_blocks else (np.arange(pairs),)
        offsets = np.cumsum([0, *(len(ids)*(len(ids)+1)//2 for ids in blocks)], dtype=np.intp)
        projected = np.zeros(offsets[-1])
        local = np.empty(pairs, dtype=np.intp)
        for ids in blocks:
            local[ids] = np.arange(len(ids))
        for (g, h), values in self.blocks.items():
            if use_blocks and g != h:
                continue
            p, q = self.pairs[g].T
            r, s = self.pairs[h].T
            a, b = np.maximum(p, q), np.maximum(r, s)
            left = local[a*(a+1)//2 + np.minimum(p, q)]
            right = local[b*(b+1)//2 + np.minimum(r, s)]
            start = offsets[g] if use_blocks else 0
            for i, row in zip(left, values):
                high = np.maximum(i, right)
                addresses = start + high*(high+1)//2 + np.minimum(i, right)
                np.add.at(projected, addresses, row)
        return projected
