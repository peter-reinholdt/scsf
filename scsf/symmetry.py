"""Real one-dimensional irreps of D2h and its subgroups."""
from dataclasses import dataclass
from operator import index


_IRREPS = {
    'D2h': ('Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u'),
    'D2': ('A', 'B1', 'B2', 'B3'),
    'C2v': ('A1', 'A2', 'B1', 'B2'),
    'C2h': ('Ag', 'Bg', 'Au', 'Bu'),
    'C2': ('A', 'B'),
    'Cs': ("A'", 'A"'),
    'Ci': ('Ag', 'Au'),
    'C1': ('A',),
}


@dataclass(frozen=True, slots=True, init=False)
class PointGroup:
    """Immutable subgroup and orbital irreps in the supplied orbital order.

    Irreps may be subgroup names or zero-based XOR codes (PySCF's numbering
    for these subgroups). The first irrep is always the identity.
    """

    group: str
    orbital_irreps: tuple[int, ...]

    def __init__(self, group, orbital_irreps):
        group = next((name for name in _IRREPS if name.lower() == str(group).lower()), None)
        if group is None:
            raise ValueError('point group must be D2h or one of its subgroups')
        object.__setattr__(self, 'group', group)
        object.__setattr__(self, 'orbital_irreps', tuple(self.irrep(value) for value in orbital_irreps))

    @property
    def names(self):
        return _IRREPS[self.group]

    def irrep(self, value):
        if isinstance(value, str):
            names = tuple(name.lower() for name in self.names)
            key = value.replace("''", '"').lower()
            if key in names:
                return names.index(key)
            raise ValueError(f'unknown {self.group} irrep {value!r}')
        code = index(value)
        if not 0 <= code < len(self.names):
            raise ValueError(f'{self.group} irrep code out of range')
        return code

    def configuration_irrep(self, occupations):
        """Return the product irrep of a spatial occupation or CSF label."""
        if len(occupations) != len(self.orbital_irreps):
            raise ValueError('occupation and orbital counts must match')
        result = 0
        for occupation, irrep in zip(occupations, self.orbital_irreps):
            if occupation in (1, 'u', 'd'):
                result ^= irrep
            elif occupation not in (0, 2, '0', '2'):
                raise ValueError('occupations must be 0, 1, 2 or CSF steps')
        return result

    def _check_size(self, size):
        if len(self.orbital_irreps) != size:
            raise ValueError('point-group metadata must give one irrep per orbital')
        return self

    def _pair_blocks(self):
        import numpy as np

        irreps = np.asarray(self.orbital_irreps, dtype=np.intp)
        p, q = np.tril_indices(len(irreps))
        labels = irreps[p] ^ irreps[q]
        return p, q, tuple(np.flatnonzero(labels == code) for code in range(len(self.names)))

    def _encode(self):
        return dict(group=self.group, orbital_irreps=self.orbital_irreps)

    @classmethod
    def _decode(cls, data):
        if not isinstance(data, dict) or set(data) != {'group', 'orbital_irreps'}:
            raise ValueError('invalid point-group metadata')
        return cls(data['group'], data['orbital_irreps'])


def check_symmetry(symmetry, size):
    if symmetry is None:
        return None
    if not isinstance(symmetry, PointGroup):
        raise TypeError('symmetry must be a PointGroup')
    return symmetry._check_size(size)
