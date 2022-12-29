from typing import Tuple, Sequence
import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X: np.ndarray, y: np.ndarray = None, features_names: Sequence[str] = None, label_names: str = None):
        """
        Construtor do objeto Dataset.

        :param X: Array com os valores das features (variável independente)
        :param y: Array que indica se existem ou não labels (variável dependente) (se existir (True), o modelo será supervisionado)
        :param features_names: Lista de strings com o nome das features
        :param label_names: String com nome das labels
        """
        if X is None:
            raise ValueError('X cannot be none')

        if features_names is None:
            features_names_ = [str(i) for i in range(X.shape[1])]

        else:
            features_names = list(features_names)

        if y is not None and label_names is None:
            label_names = 'y'

        self.X = X
        self.y = y
        self.features_names = features_names
        self.label_names = label_names

    def shape(self) -> Tuple[int, int]:
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
        return pd.DataFrame(self.X, columns=self.features_names, index=self.y)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame
        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features_names=features, label_names=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame
        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features_names=features, label_names=label)

if __name__ == '__main__':
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features_names=features, label_names=label) # S
    dataset_naosuperv = Dataset(X=x, y=None, features_names=features, label_names=label) # NS
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
    dataset_null = Dataset(X=x, y=y, features_names=features, label_names=label)

    print("Imprimir dataset:")
    print(dataset_null.print_dataframe())
    # Removing NaN:
    dataset_null.drop_null()
    print("Dataset depois de drop_null:")
    print(dataset_null.X) # print_dataframe not working
    # Filling NaN:
    dataset_null = Dataset(X=x, y=y, features_names=features, label_names=label)
    dataset_null.replace_null(10)
    print("Dataset depois de replace_null:")
    print(dataset_null.print_dataframe())
