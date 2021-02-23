from collections import defaultdict

from ase.neighborlist import natural_cutoffs
from ase.neighborlist import build_neighbor_list


class NeighborList:
    def __init__(self):
        self.list = defaultdict(lambda: set())

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value

    def construct(self, atoms, scale_factor=1.0):
        neighbor_list = build_neighbor_list(atoms,
                                            cutoffs=natural_cutoffs(atoms, mult=scale_factor),
                                            bothways=True,
                                            self_interaction=False)

        for atom_idx, _ in enumerate(atoms):
            neighbors, _ = neighbor_list.get_neighbors(atom_idx)
            self.list[atom_idx] = set(neighbors)

    def get_coordination_number(self, atom_idx):
        return len(self.list[atom_idx])

    def get_n_bonds(self):
        n_bonds = sum([len(l) for l in list(self.list.values())])
        return n_bonds/2
