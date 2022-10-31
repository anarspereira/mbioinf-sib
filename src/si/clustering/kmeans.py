from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance


class KMeans:

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        """
        Construtor do K-Means.

        :param k: Inteiro. Nº de centróides.
        :param max_iter: Inteiro. Nº de iterações.
        :param distance: Euclidean distance
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
        :return:
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[seeds] # se k = 2, tem dois vetores

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Método que procura o centróide com distância mais curta de cada ponto dos dados.
        Get the closest centroid to each data point.

        :param sample: np.ndarray. shape(n_features,). Uma amostra.

        :return: ndarray do centróide com distância mais curta de cada ponto.
        """
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis = 0) # 0 porque só temos um vetor em linhas e vai buscar o index da menor distância
        return closest_centroid_index

    def fit(self):
        """
        Método que calcula a distância entre uma amostra e os vários centróides do dataset.

        np.random.permutation - cria um vetor aleatório que pode ser usado para selecionar amostras do dataset

        :return: K centróides
        """
        # fitting the k-means
        convergence = False
        i = 0
        labels = np.zeros(dataset.shape()[0])
        while not convergence and i = self.max_iter:
            # get closest centroids
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis = 1, arr = dataset.X)

            # compute the new centroids
            centroids = []
            for j in range(self.k):
                centroid = np.mean(dataset.X[new_labels == j], axis = 0) # dá uma boolean mask que vai dizer que, se i=0, vai ser true e false num array (labels) da mesma dimensão de x
                centroids.append(centroid)
            self.centroids = np.array(centroids)

            # check if the centroids have changed
            convergence = np.any(new_labels != labels)

            # replace labels
            #.........

    def _get_distances(self):
        pass

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Metodo que transforma o dataset.
        Método que calcula a distância entre cada amostra e o centróide mais próximo.

        :param dataset: Objeto dataset.

        :return: ndarray com as distâncias.
        """
        centroids_distances = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.X)
        return centroids_distances

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Método que prevê as labels do dataset.

        :param dataset: Objetivo dataset.

        :return: Predicted labels.
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)