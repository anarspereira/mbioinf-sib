from typing import Tuple

import numpy as np

from si.data.dataset_module import Dataset


def train_test_split (dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Método para dividir um dataset em dataset de treino e de teste.

    :param dataset: Dataset para dividir em treino e teste.
    :param test_size: Tamanho do dataset do teste.
    :param random_state: Seed para gerar permutações.

    :return: Tuplo com dataset de treino e dataset de teste.
    """
    # set random state
    np.random.seed(random_state) # adicionar sempre isto para não obter coisas aleatórias e poder reproduzir esta pipeline vezes sem conta sempre com o mesmo resultado

    # get dataset size
    n_samples = dataset.shape()[0] # ir buscar o nº de amostras

    # get number of samples in the test set
    n_test = int(n_samples * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # get the samples in the test set
    test_idxs = permutations[:n_test] # até n_test

    # get samples in the training set
    train_idxs = permutations[n_test:] # depois de n_test até ao fim

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test