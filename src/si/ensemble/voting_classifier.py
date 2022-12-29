import numpy as np

from si.data.dataset_module import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:

    def __init__(self, models: list):
        """

        :param models: Lista de conjunto de modelos
        """
        # parameters
        self.models = models
        pass

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: VotingClassifier
            The fitted model
        """
        for model in self.models:
            node.fit(dataset)
        return self

    # helper function - não precisa de estar aqui. está dentro porque não é mais precisa noutro lado
    def _get_majority_vote(pred: np.ndarray) -> int:
        # get the most comon label
        labels, counts = np.unique(pred, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset.

        Vai modelo a modelo, eleitor a eleitor, perguntar "quem achas que vai ganhar a eleição".
        Pega neles todos, escolhe o que tem maior contagem e é isso que ele retorna.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # caso não fizer a transposta, posso colocar axis=0
        return np.apply_along_axis(self._get_majority_vote, axis=1, arr=predictions) # aplicar o método linha por linha (axis=1)

    def score(self, dataset: Dataset) -> float:
        """
        Método com a função accuracy, que retorna a accuracy do modelo e o dado dataset.

        :param dataset: Dataset de teste (y_true) (y são as labels).

        :return: Valor de accuracy do modelo.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)