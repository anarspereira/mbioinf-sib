import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X: np.ndarray, y: np.ndarray, features: list, label: str):
        #TODO: não está a assumir np.ndarray, tentar resolver
        """
        Construtor.

        :param X: Array com valores das features (variáveis independentes) - np.ndarray
        :param y: Array com bool se a variável dependente é supervisionada ou não (True ou False) (np.udarray?)
        :param features: Lista de strings com o nome das features
        :param label: String com nome do vetor da variável dependente
        """
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> tuple:
        """
        Dá a forma do dataset.

        :return: Tuplo com nº de exemplos e nº de colunas.
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Verifica se o dataset é supervisionado (tem label (vetor y)) ou não supervisionado (não tem label).

        :return: Booleano: true se tem label; false se não tem label.
        """
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self) -> list:
        #TODO: descobrir se devo dar raise de uma exception ou typeerror
        #TODO: descobrir se é possível dar raise sem parar o programa
        """
        Classes do dataset.

        :return: Lista com os valores únicos.
        """
        if self.y is None:
            raise Exception('Dataset não supervisionado (sem label).')
        else:
            return np.unique(self.y)

    def get_mean(self) -> list:
        """
        Calcula a média.

        :return: Lista com as médias das features.
        """
        return np.nanmean(self.X, axis=0)  # axis 0: colunas, axis 1: exemplos

    def get_variance(self) -> list:
        """
        Calcula a variância.

        :return: Lista com as variâncias das features.
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> list:
        """
        Calcula a mediana.

        :return: Lista com as medianas das features.
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> list:
        """
        Calcula o mínimo.

        :return: Lista com os valores mínimos das features.
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> list:
        """
        Calcula o máximo.

        :return: Lista com os valores máximos das features.
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Cria pandas dataframe com média, mediana, variância, mínimo e máximo das features.

        :return: Pandas dataframe.
        """
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'median': self.get_median(),
             'variance': self.get_variance(),
             'min': self.get_min(),
             'max': self.get_max()}
        )

    def remove_null(self) -> pd.DataFrame:
        """
        Remove os valores nulos.

        :return: Pandas dataframe.
        """
        if self.X is None:
            return pd.DataFrame(self.X).dropna(axis=0)

    def replace_null(self, value) -> pd.DataFrame:
        """
        Substitui os valores nulos.

        :param value: Valor que vai substituir o valor nulo.

        :return: Pandas dataframe.
        """
        if self.X is None:
            return pd.DataFrame(self.X).fillna(value)


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=label)
    dataset_naosuperv = Dataset(X=x, y=None, features=features, label=label)
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
    #TODO: testes