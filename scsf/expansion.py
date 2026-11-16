"""Data-only determinant exports for small CSF phase and energy checks."""
import numpy as np


class DeterminantExpansion:
    """Packed alpha/beta bit strings; deliberately provides no CI backend."""
    def __init__(self, nbasis, nocc_up, nocc_dn, words):
        self.nbasis, self.nocc_up, self.nocc_dn = nbasis, nocc_up, nocc_dn
        self._words = np.array(words, dtype=np.uint64, copy=True, order='C')
        if self._words.ndim != 3 or self._words.shape[1:] != (2, (nbasis+63)//64):
            raise ValueError('invalid determinant export shape')
        self._words.flags.writeable = False

    def __len__(self):
        return len(self._words)

    def to_det_array(self):
        return self._words.copy()
