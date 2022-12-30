from ctypes import Union

import numpy as np

from si.data.dataset_module import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
    def __init__(self, k: int, distance: euclidean_distance = euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        """
        Stores the dataset.
        :param dataset: Dataset object
        :return: The dataset
        """
        self.dataset = dataset
        return self

    def _get_closet_label(self, x: np.ndarray):
        """
        Calculates the class with the highest frequency.
        :param x: Array of samples.
        :return: Indexes of the classes with the highest frequency
        """

        # Calculates the distance between the samples and the dataset
        distances = self.distance(x, self.dataset.X)

        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
        knn_labels = self.dataset.y[knn]

        # Computes the mean of the matching classes
        match_class_mean = np.mean(knn_labels)

        # Sorts the classes with the highest mean
        high_freq_class = np.argsort(match_class_mean)[:self.k]

        return high_freq_class

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class with the highest frequency
        :return: Class with the highest frequency.
        """
        return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset)

        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset=dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
