from typing import Callable

import numpy as np

from si.data.dataset_module import Dataset
from si.statistics.euclidean_distance import euclidean_distance


class KMeans:
    """
    K-Means clustering no dataset.
    Agrupa as amostras em k clusters ao tentar minimizar a distância entre samples e o seu centróide mais próximo.
    Retorna os centróides e os índices do centróide mais próximo para cada ponto.

    Parâmetros:
    ----------
    k: int
        Número de clusters.
    max_iter: int
        Número máximo de iterações.
    distance: Callable
        Função que calcula a distância (default: euclidean distance).

    Atributos
    ----------
    centroids: np.array
        Centróides dos clusters.
    labels: np.array
        Labels dos clusters.
    """

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        """
        Construtor do K-Means.

        :param k: Inteiro. Nº de clusters.
        :param max_iter: Inteiro. Nº máximo de iterações.
        :param distance: Função da distâcia (default: euclidean distance)
        """
        # Parâmetros:
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        # Atributos:
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        Método que gera k centróides iniciais.

        :param dataset: Objeto. Dataset.
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[seeds] # se k = 2, tem dois vetores

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Método que procura o centróide com distância mais curta de cada ponto dos dados.

        :param sample: np.ndarray. shape(n_features,). Uma amostra.

        :return: np.ndarray do centróide com distância mais curta de cada ponto.
        """
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis=0) # 0 porque só temos um vetor em linhas e vai
        # buscar o index da menor distância
        return closest_centroid_index

    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        Método que calcula a distância entre uma amostra e os vários centróides do dataset.
        Faz o fit do clustering por k-means no dataset.
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.

        np.random.permutation - cria um vetor aleatório que pode ser usado para selecionar amostras do dataset

        :param dataset: Dataset

        :return: Objeto KMeans
        """
        # generate initial centroids
        self._init_centroids(dataset)

        # fitting the k-means
        convergence = False
        i = 0
        labels = np.zeros(dataset.shape()[0])
        while not convergence and i < self.max_iter:

            # get closest centroid
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

            # compute the new centroids
            centroids = []
            for j in range(self.k):
                centroid = np.mean(dataset.X[new_labels == j], axis=0) # dá uma boolean mask que vai dizer que, se i=0,
                # vai ser true e false num array (labels) da mesma dimensão de x
                centroids.append(centroid)

            self.centroids = np.array(centroids)

            # check if the centroids have changed
            convergence = np.any(new_labels != labels)

            # replace labels
            labels = new_labels

            # increment counting
            i += 1

        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        It computes the distance between each sample and the closest centroid.

        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.

        Returns
        -------
        np.ndarray
            Distances between each sample and the closest centroid.
        """
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Metodo que transforma o dataset.
        Método que calcula a distância entre cada amostra e o centróide mais próximo.

        :param dataset: Objeto dataset.

        :return: ndarray transformado com as distâncias.
        """
        centroids_distances = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.X)
        return centroids_distances

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Método que prevê as labels do dataset.

        :param dataset: Objetivo dataset.

        :return: Predicted labels.
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and predicts the labels of the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        self.fit(dataset)
        return self.predict(dataset)


if __name__ == '__main__':
    dataset_ = Dataset.from_random(100, 5)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)
