from typing import Union, Callable

import numpy as np

from si.data.dataset_module import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Construtor da class KNNClassifier.

        :param k: Número de k exemplos de nearest neighbors a considerar.
        :param distance: Função que calcula a distância entre a amostra e as amostras do dataset de treino
        """
        # Parâmetros
        self.k = k
        self.distance = euclidean_distance # euclidean distance

        # Atributos
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Método para fazer o fit do modelo de acordo com o input dataset.

        :param dataset: Dataset de treino.

        :return: self. O modelo treinado
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        Método que retorna a label mais próxima de uma dada sample.

        :param sample: Array com a sample.

        :return: String ou inteiro com a label mais próxima.
        """
        # passo 1 - calcular distância usando a distância euclideana - compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # passo 2 - ver quais estão mais perto desta amostra - get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # passo 3 - selecionar as classes desses exemplos - get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # passo 4 - get the most common label
        # obter classe mais comum nos k exemplos através do np.unique, que retornará um array que diz as contagens de
        # cada um dos valores do y_prep e outro array com os valores únicos que estão ali depois, para selecionar o mais
        # comum, selecionar o maior através de np.argmax
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

        :param dataset: Dataset de teste (y_true) (y são as labels) onde avaliar o modelo.

        :return: Valor de accuracy do modelo.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset=dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
