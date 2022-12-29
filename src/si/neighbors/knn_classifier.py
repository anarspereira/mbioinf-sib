from typing import Union

import numpy as np

from si.data.dataset_module import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier:

    def __init__(self, k, distance, dataset):
        """
        Construtor da class KNNClassifier.

        :param k: Número de k exemplos a considerar.
        :param distance: Função que calcula a distância entre a amostra e as amostras do dataset de treino
        """
        self.k = k
        self.distance = euclidean_distance # euclidean distance
        self.dataset = dataset

    def fit(self, dataset: Dataset) -> Dataset:
        """
        Método para armazenar o dataset de treino.

        :param dataset: Dataset de treino.

        :return: O próprio dataset.
        """
        self.dataset = dataset

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        Método que retorna a label mais próxima de uma dada sample.

        :param sample: Array com a sample.

        :return: String ou inteiro com a label mais próxima..
        """
        # passo 1 - calcular distância usando a distância euclideana - compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # passo 2 - ver quais estão mais perto desta amostra - get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # passo 3 - selecionar as classes desses exemplos - get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # passo 4 - get the most common label
        # obter classe mais comum nos k exemplos através do np.unique, que retornará um array que diz as contagens de cada um dos valores do y_prep e outro array com os valores únicos que estão ali
        # depois, para selecionar o mais comum, selecionar o maior através de np.argmax
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        # array das labels - binário
        # array counts - contagem de labels
        return labels[np.argmax(counts)]  # vou buscar às labels qual o index que tem mais contagens

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Método para prever as classes de um dado dataset.

        :param dataset: Dataset de teste.

        :return: Array com as previsões das classes para o dataset de teste.
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Método com a função accuracy, que retorna a accuracy do modelo e o dado dataset.

        :param dataset: Dataset de teste (y_true) (y são as labels).

        :return: Valor de accuracy do modelo.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)