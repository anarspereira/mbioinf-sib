from typing import Sequence

import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X: np.ndarray, y: np.ndarray, features: list, label: str):
        """
        Construtor do objeto Dataset.

        :param X: Array com os valores das features (variável independente)
        :param y: Array que indica se existem ou não labels (variável dependente) (se existir (True), o modelo será supervisionado)
        :param features: Lista de strings com o nome das features
        :param label: String com nome das labels
        """
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> tuple:
        """
        Dá a forma do dataset.

        :return: Tuplo com nº de exemplos/observações e nº de features (colunas).
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Verifica se o dataset é supervisionado (tem labels (vetor y)) ou não supervisionado (não tem labels).

        :return: Booleano: True se tem label; False se não tem label.
        """
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self) -> list:
        """
        Retorna as classes do dataset.

        :return: Lista com os valores únicos do dataset.
        """
        if self.y is None:
            raise ValueError('Dataset não supervisionado (sem label).')
        else:
            return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """
        Calcula a média de cada feature.

        :return: Array com as médias das features.
        """
        return np.nanmean(self.X, axis=0)  # axis 0: features (colunas), axis 1: exemplos/observações (linhas)

    def get_variance(self) -> np.ndarray:
        """
        Calcula a variância de cada feature.

        :return: Array com as variâncias das features.
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Calcula a mediana de cada feature.

        :return: Array com as medianas das features.
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Calcula o mínimo.

        :return: Lista com os valores mínimos das features.
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Calcula o máximo de cada feature.

        :return: Lista com os valores máximos das features.
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Cria um dataframe com os valores do summary (média, mediana, variância, mínimo e máximo) das features.

        :return: Pandas dataframe com o summary das features.
        """
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'median': self.get_median(),
             'variance': self.get_variance(),
             'min': self.get_min(),
             'max': self.get_max()}
        )

    def drop_null(self):
        """
        Remove observações que contenham pelo menos um valor nulo (NaN).
        """
        self.X = self.X[~np.isnan(self.X).any(axis=1)]
        return self.X

    def replace_null(self, value):
        """
        Substitui os valores nulos.

        :param value: Valor que vai substituir o valor nulo.
        """
        self.X = np.nan_to_num(self.X, nan=value)
        return self.X

    def print_dataframe(self) -> pd.DataFrame:
        """
        Imprime o dataframe.

        :return: Dataframe
        """
        return pd.DataFrame(self.X, columns=self.features, index=self.y)

if __name__ == '__main__':
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=label) # S
    dataset_naosuperv = Dataset(X=x, y=None, features=features, label=label) # NS
    print("[S] Shape: ", dataset.shape())
    print("[S] É supervisionado: ", dataset.has_label())
    print("[NS] É supervisionado: ", dataset_naosuperv.has_label())
    print("[S] Classes: ", dataset.get_classes())
    # print(dataset.get_mean())
    # print(dataset.get_variance())
    # print(dataset.get_median())
    # print(dataset.get_min())
    # print(dataset.get_max())
    print("[S] Summary:\n", dataset.summary())

    # Removing and replacing NaN:
    x = np.array([[1, 2, 3],
                  [1, np.nan, 3],
                  [1, 2, np.nan],
                  [np.nan, 2, 3]])
    y = np.array([1, 2, 4, 4])
    dataset_null = Dataset(X=x, y=y, features=features, label=label)

    print("Imprimir dataset:")
    print(dataset_null.print_dataframe())
    # Removing NaN:
    dataset_null.drop_null()
    print("Dataset depois de drop_null:")
    print(dataset_null.X) # print_dataframe not working
    # Filling NaN:
    dataset_null = Dataset(X=x, y=y, features=features, label=label)
    dataset_null.replace_null(10)
    print("Dataset depois de replace_null:")
    print(dataset_null.print_dataframe())
