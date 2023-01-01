import numpy as np

from si.data.dataset_module import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.
    """

    def __init__(self, models: list):
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        """
        # parameters
        self.models = models

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
            model.fit(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset - predict class labels for samples in X.

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
        def _get_majority_vote(pred: np.ndarray) -> int: # helper function - não precisa de estar aqui. está dentro
            # porque não é mais precisa noutro lado
            # get the most comon label
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # caso não fizer a transposta, posso colocar axis=0
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions) # aplicar o método linha por linha
        # (axis=1)

    def score(self, dataset: Dataset) -> float:
        """
        Método com a função accuracy, que retorna a accuracy do modelo e o dado dataset.

        :param dataset: Dataset de teste (y_true) (y são as labels).

        :return: Valor de accuracy do modelo.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    # imports
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))
