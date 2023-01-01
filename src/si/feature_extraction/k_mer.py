import itertools

import numpy as np

from si.data.dataset_module import Dataset


class KMer:
    """
    A sequence descriptor that returns the k-mer composition of the sequence.
    Parameters
    ----------
    k : int
        The k-mer length.
    Attributes
    ----------
    k_mers : list of str
        The k-mers.
    """
    def __init__(self, k: int = 3, alphabet: str = 'DNA'):
        """
        Construtor da classe KMer. Esta classe é específica para DNA (alfabeto: ACTG).

        :param k: Tamanho da substring
        :param alphabet: Alfabeto da sequência (default: DNA)
        """
        # Parâmetros
        self.k = k
        self.alphabet = alphabet.upper()

        if self.alphabet == 'DNA':
            self.alphabet = 'ACTG'
        elif self.alphabet == 'PROT':
            self.alphabet = 'ACDEFGHIKLMNPQRSTVWY_'

        # Atributos
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        Método para estimar todos os k-mers possíveis.

        :param dataset: The dataset to fit the model to.

        :return: self: KMer. Retorna o fitted descriptor.
        """
        # generate possible k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]  # itera o tuplo e
        # adiciona-o à string. se o tuplo tiver ACTG, ele adiciona a string vazia e vamos ficar com uma string ACTG

        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Método que calcula a frequência de cada k-mer em cada sequência do dataset

        :param sequence: Sequência

        :return: Array do dataset
        """
        # calculate the k-mer composition - todos os k-mers possíveis
        kmer_count = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            kmer_count[k_mer] += 1

        # normalize the counts
        return np.array([kmer_count[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Calcula a composição do k-mer dada uma determinada sequência.

        :return: Dataset
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features_names=self.k_mers, label_names=dataset.label_names)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Método para correr o fit e o transform.

        :return: Dataset transformado
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset_module import Dataset

    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features_names=['sequence'],
                       label_names='label')

    k_mer_ = KMer(k=2, alphabet="DNA")
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features_names)