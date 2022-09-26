import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X: object, y: object, features: list, label: str) -> object:
        """
        Construtor.
        :param X: Numpy np array - array com valores
        :param y: Numpy ud array - array com bool se é supervisionado ou não (True ou False)
        :param features: Lista de strings com features
        :param label: String
        """
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape (self):
        """
        Dá a forma do dataset.
        :return: Tuplo com nº de exemplos e nº de colunas.
        """
        return self.X.shape

    def has_label (self):
        """
        Verifica se o dataset é supervisionado (tem label (vetor y)) ou não supervisionado (não tem label).
        :return: True se tem label; False se não tem label.
        """
        if self.y is not None:
            return True
        else:
            return False

    def get_classes (self):
        """
        Classes do dataset.
        :return: Valores únicos.
        """
        if self.y is None:
            return

        return np.unique(self.y)

    def get_mean (self):
        """
        Calcula a média.
        :return: Média.
        """
        return np.nanmean(self.X, axis = 0) # axis 0: colunas, axis 1: exemplos

    def get_variance (self):
        """
        Calcula a variância.
        :return: Variância.
        """
        return np.nanvar(self.X, axis = 0)

    def get_median (self):
        """
        Calcula a mediana.
        :return: Mediana.
        """
        return np.nanmedian(self.X, axis = 0)

    def get_min (self):
        """
        Calcula o mínimo.
        :return: Mínimo.
        """
        return np.nanmin(self.X, axis = 0)

    def get_max (self):
        """
        Calcula o máximo.
        :return: Máximo.
        """
        return np.nanmax(self.X, axis = 0)

    def summary (self):
        """
        Cria um pandas dataframe com média, mediana, mínimo e máximo.
        :return: Pandas dataframe.
        """
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()}
        )

if __name__ == '__main__':
    x = np.array([[1,2,3], [1,2,3]])
    y = np.array([1,2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X = x, y = y, features = features, label = label)
    dataset_naosuperv = Dataset(X = x, y = None, features = features, label = label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset_naosuperv.has_label())
    print(dataset.get_classes())
    print(dataset_naosuperv.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())