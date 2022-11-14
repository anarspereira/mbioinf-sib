import itertools

import numpy as np

from si.data.dataset import Dataset


class KMer:

    def __init__(self, k: int = 3, k_mers):
        """
        Construtor da classe KMer. Esta classe é específica para DNA (alfabeto: ACTG).

        :param k: Tamanho da substring
        :param k_mers: Todos os k-mers possíveis
        """
        # Parâmetros
        self.k = k

        # Atributos
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        Método para estimar todos os k-mers possíveis.

        :param dataset: The dataset to fit the model to.

        :return: self: KMer. Retorna o fitted descriptor.
        """
        # generate the k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product('ACTG', repeat = self.k)] # itera o tuplo e adiciona-o à string. se o tuplo tiver ACTG, ele adiciona a string vazia e vamos ficar com uma string ACTG
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """

        :param sequence:

        :return:
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self) -> Dataset:
        """

        :return:
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
                                       for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset)

    def fit_transform(self):
        """

        :return:
        """
        return self.fit(dataset).transform(dataset)