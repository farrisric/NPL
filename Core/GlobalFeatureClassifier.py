import numpy as np
import copy


class GlobalFeatureClassifier:
    """Base class for classifiers that produce a feature vector directly from a particle without the need
    for precomputing local environments.

    Valid implementations have to implement the compute_feature_vector(particle) method.
    """
    def __init__(self):
        self.feature_key = None

    def get_feature_key(self):
        return self.feature_key

    def set_feature_key(self, feature_key):
        self.feature_key = feature_key

    def compute_feature_vector(self, particle):
        raise NotImplementedError


# TODO better name
# TODO refactor handling of symbols
class SimpleFeatureClassifier(GlobalFeatureClassifier):
    """Base class for classifiers which provides handling of two elements and calculation of bond counts.

    Entries for different elements in feature vectors have to be ordered consistently. This class uses an
     alphabetical ordering of TWO elements. The returned feature vector consists of [n_aa_bonds/n_atoms,
     n_ab_bonds/n_atoms, n_bb_bonds/n_atoms, n_a_atoms*0.1].
    """
    def __init__(self, symbols):
        GlobalFeatureClassifier.__init__(self)
        symbols_copy = copy.deepcopy(symbols)
        symbols_copy.sort()
        self.symbol_a = symbols_copy[0]
        self.symbol_b = symbols_copy[1]

        self.feature_key = 'SFC'
        return

    def compute_feature_vector(self, particle):
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        n_atoms = particle.atoms.get_n_atoms()

        M = particle.get_stoichiometry()[self.symbol_a] * 0.1
        particle.set_feature_vector(self.feature_key, np.array([n_aa_bonds / n_atoms, n_bb_bonds / n_atoms, n_ab_bonds / n_atoms, M]))

    def compute_respective_bond_counts(self, particle):
        n_aa_bonds = 0
        n_ab_bonds = 0
        n_bb_bonds = 0

        for lattice_index_with_symbol_a in particle.atoms.get_indices_by_symbol(self.symbol_a):
            neighbor_list = particle.neighbor_list[lattice_index_with_symbol_a]
            for neighbor in neighbor_list:
                symbol_neighbor = particle.atoms.get_symbol(neighbor)

                if self.symbol_a != symbol_neighbor:
                    n_ab_bonds += 0.5
                else:
                    n_aa_bonds += 0.5

        for lattice_index_with_symbol_b in particle.atoms.get_indices_by_symbol(self.symbol_b):
            neighbor_list = particle.neighbor_list[lattice_index_with_symbol_b]
            for neighbor in neighbor_list:
                symbol_neighbor = particle.atoms.get_symbol(neighbor)

                if self.symbol_b == symbol_neighbor:
                    n_bb_bonds += 0.5
                else:
                    n_ab_bonds += 0.5

        return n_aa_bonds, n_bb_bonds, n_ab_bonds


class TopologicalFeatureClassifier(SimpleFeatureClassifier):
    """Classifier for a generalization of the topological descriptors by Kozlov et al. (2015).

    Implemented for TWO elements, which are sorted alphabetically.The returned feature vector will have the form
    [n_aa_bonds/n_atoms, n_ab_bonds/n_atoms, n_bb_bonds/n_atoms, n_a_atoms*0.1, n_a(cn=0), n_a(cn=1), ..., n_a(cn=12].
    """
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'TFC'

    def compute_feature_vector(self, particle):
        n_atoms = particle.get_n_atoms()
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        coordinated_atoms = [len(particle.get_atom_indices_from_coordination_number([cn], symbol=self.symbol_a)) for cn in range(13)]

        M = particle.get_stoichiometry()[self.symbol_a]*0.1

        feature_vector = np.array([n_aa_bonds/n_atoms, n_bb_bonds/n_atoms, n_ab_bonds/n_atoms, M] + coordinated_atoms)
        particle.set_feature_vector(self.feature_key, feature_vector)


# TODO move to separate modules & rename
class CoordinationFeatureClassifier(SimpleFeatureClassifier):
    """Feature classifier that counts how often each coordination number appears if vacancies (atoms of element 'X')
     are present in an otherwise PURE particle.

     Form of the feature vector:
     [n_aa_bonds, n_a(cn=0), n_a(cn=1), ... n_a(cn=12)]
     """
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'TFC'

    def compute_feature_vector(self, particle):
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        symbol_a_features = [0]*13
        for index in particle.get_indices():
            if particle.get_symbol(index) != 'X':
                env = particle.get_local_environment(index)
                symbol_a_features[env[0]] += 1

        features = [n_aa_bonds] + symbol_a_features
        feature_vector = np.array(features)
        particle.set_feature_vector(self.feature_key, feature_vector)
